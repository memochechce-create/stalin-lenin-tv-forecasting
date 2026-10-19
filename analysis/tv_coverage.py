"""Read-only archive-record coverage; HTTP inventories are UTC, output days MSK.

load_coverage(path, start, end) returns JSON-serializable daily/broadcast rows,
canonical_id_by_id and diagnostics. Dates are inclusive ISO strings or dates.
No transcript files, exports or database records are modified.
"""
import datetime as dt
import json
from collections import Counter, defaultdict

try:
    from .source_database import open_source_database
except ImportError:
    from source_database import open_source_database

UTC = dt.timezone.utc
MSK = dt.timezone(dt.timedelta(hours=3), 'MSK')
CHANNELS = ('1TV', 'NTV', 'RUSSIA1', 'RUSSIA24')


def _date(value):
    return dt.date.fromisoformat(value) if isinstance(value, str) else value


def _timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _runtime(value):
    try:
        h, m, s = map(float, value.split(':'))
        return h * 3600 + m * 60 + s
    except (AttributeError, TypeError, ValueError):
        return None


def _union_seconds(intervals):
    total, latest = 0.0, None
    for start, stop in sorted(intervals):
        if latest is None or start > latest:
            total += (stop - start).total_seconds()
        elif stop > latest:
            total += (stop - latest).total_seconds()
        latest = max(latest, stop) if latest is not None else stop
    return total


def load_coverage(db, start, end, supplemental=None):
    """Build conservative coverage without claiming complete television airtime.

    Canonicalization applies only to identical (channel, UTC start) groups:
    successful transcript first, longest interval, most words, named programme,
    then lexical ID. Different starts, repeated programmes and minute overlaps
    remain separate. raw_* always describes every listed archive ID.
    """
    start, end = _date(start), _date(end)
    if start > end:
        raise ValueError('start must not be after end')
    with open_source_database(db, supplemental) as connection:
        inventories = {(r['channel'], r['day'].replace('-', '')): str(r['status'])
                       for r in connection.execute('SELECT channel,day,status FROM inventory')}
        channels = set(CHANNELS) | {c for c, _ in inventories}
        rows, invalid = [], []
        for source in connection.execute('SELECT id,channel,day,metadata,status,word_count FROM broadcast ORDER BY id'):
            record = dict(source)
            metadata = json.loads(record.pop('metadata'))
            try:
                began, stopped = _timestamp(metadata['start_time']), _timestamp(metadata['stop_time'])
            except (ValueError, KeyError, TypeError) as exc:
                invalid.append({'id': record['id'], 'error': str(exc)})
                continue
            local_day = began.astimezone(MSK).date()
            # One preceding MSK day supplies overlap context for a subset load.
            # It is removed from returned rows/denominators below.
            if not start-dt.timedelta(days=1) <= local_day <= end:
                continue
            seconds = (stopped - began).total_seconds()
            words = record['word_count']
            available = str(record['status']) == '200'
            flags = []
            if seconds <= 0:
                flags.append('invalid_duration')
            if available and words is None:
                flags.append('missing_word_count')
            if available and words is not None and words < 20:
                flags.append('short_asr')
                if seconds > 600:
                    flags.append('long_low_word_asr')
            runtime = _runtime(metadata.get('runtime'))
            record.update(day=local_day.isoformat(), inventory_utc_day=record['day'],
                          start_time=began.isoformat(), stop_time=stopped.isoformat(),
                          duration_seconds=seconds, runtime_seconds=runtime,
                          runtime_difference_seconds=(seconds-runtime if runtime is not None else None),
                          words_per_minute=(words*60/seconds if available and words is not None and seconds > 0 else None),
                          available=available, program=metadata.get('program', ''),
                          quality_flags=flags, overlap_previous_seconds=0.0)
            if metadata.get('start_localtime'):
                reported = dt.datetime.fromisoformat(metadata['start_localtime']).replace(tzinfo=MSK)
                if reported.astimezone(UTC) != began:
                    flags.append('local_time_mismatch')
            rows.append(record)
    groups = defaultdict(list)
    for row in rows:
        groups[row['channel'], row['start_time']].append(row)
    mapping = {}
    for group in groups.values():
        canonical = min(group, key=lambda r: (-int(r['available']), -r['duration_seconds'],
                        -(r['word_count'] or 0), -int(bool(r['program'])), r['id']))
        for row in group:
            mapping[row['id']] = canonical['id']
            row.update(canonical_id=canonical['id'], is_canonical=row is canonical,
                       same_start_group_size=len(group))
    by_channel = defaultdict(list)
    for row in rows:
        if row['is_canonical']:
            by_channel[row['channel']].append(row)
    for channel_rows in by_channel.values():
        latest_stop = None
        for row in sorted(channel_rows, key=lambda r: (r['start_time'], r['id'])):
            began, stopped = _timestamp(row['start_time']), _timestamp(row['stop_time'])
            if latest_stop is not None and latest_stop > began:
                row['overlap_previous_seconds'] = max(0, (min(latest_stop, stopped)-began).total_seconds())
                row['quality_flags'].append('overlapping_archive_record')
            latest_stop = max(latest_stop, stopped) if latest_stop else stopped

    rows = [r for r in rows if start.isoformat() <= r['day'] <= end.isoformat()]
    mapping = {r['id']: r['canonical_id'] for r in rows}
    groups = {key: group for key, group in groups.items() if group[0]['id'] in mapping}
    overlap_count = sum(r['is_canonical'] and r['overlap_previous_seconds'] > 0 for r in rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['channel'], row['day']].append(row)
    daily = []
    for offset in range((end-start).days+1):
        day = start + dt.timedelta(days=offset)
        for channel in sorted(channels):
            raw = grouped[channel, day.isoformat()]
            canonical = [r for r in raw if r['is_canonical']]
            previous = inventories.get((channel, (day-dt.timedelta(days=1)).strftime('%Y%m%d')), 'not_collected')
            current = inventories.get((channel, day.strftime('%Y%m%d')), 'not_collected')
            reasons = []
            if previous != '200': reasons.append('previous_utc_inventory_unavailable')
            if current != '200': reasons.append('current_utc_inventory_unavailable')
            if not canonical: reasons.append('no_listed_broadcasts')
            if any(not r['available'] for r in canonical): reasons.append('missing_transcripts')
            if any('long_low_word_asr' in r['quality_flags'] for r in canonical): reasons.append('long_low_word_asr')
            if any(any(f in r['quality_flags'] for f in ('invalid_duration', 'missing_word_count', 'local_time_mismatch')) for r in canonical):
                reasons.append('invalid_metadata')
            result = dict(channel=channel, day=day.isoformat(), previous_utc_inventory_status=previous,
                          current_utc_inventory_status=current, inventories_complete=previous == current == '200',
                          observed=not reasons, coverage_status=('observed_listed_broadcasts' if not reasons else 'incomplete'),
                          coverage_reasons=reasons, same_start_extra_records=len(raw)-len(canonical),
                          overlapping_broadcasts=sum(r['overlap_previous_seconds'] > 0 for r in canonical))
            for prefix, records in (('raw', raw), ('canonical', canonical)):
                available = [r for r in records if r['available']]
                result.update({prefix+'_listed_broadcasts': len(records), prefix+'_available_broadcasts': len(available),
                               prefix+'_missing_transcripts': len(records)-len(available),
                               prefix+'_word_count': sum(r['word_count'] or 0 for r in available),
                               prefix+'_available_duration_seconds': sum(max(0, r['duration_seconds']) for r in available),
                               prefix+'_short_asr_broadcasts': sum('short_asr' in r['quality_flags'] for r in available)})
            midnight = dt.datetime.combine(day, dt.time(), MSK).astimezone(UTC)
            tomorrow = midnight+dt.timedelta(days=1)
            result['canonical_available_union_seconds_started_today'] = _union_seconds([
                (max(midnight, _timestamp(r['start_time'])), min(tomorrow, _timestamp(r['stop_time'])))
                for r in canonical if r['available'] and r['duration_seconds'] > 0])
            daily.append(result)
    return dict(daily=daily, broadcasts=rows, canonical_id_by_id=mapping,
                diagnostics=dict(invalid_timestamp_records=invalid, same_start_groups=sum(len(g)>1 for g in groups.values()),
                                 same_start_extra_records=len(rows)-len(groups), overlapping_canonical_records=overlap_count,
                                 quality_flags=dict(Counter(f for r in rows for f in r['quality_flags'])),
                                 unit='archive_record_by_MSK_start_day', day_timezone='UTC+03:00',
                                 overlap_policy='flag_only_no_cue_or_mention_deduplication',
                                 observed_scope='all_canonical_listed_transcripts_available_with_conservative_quality_checks'))
