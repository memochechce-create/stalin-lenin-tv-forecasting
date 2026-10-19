import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from analysis.tv_coverage import load_coverage


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'tv.sqlite'
        self.db = sqlite3.connect(self.path)
        self.addCleanup(self.db.close)
        self.db.executescript('CREATE TABLE inventory(channel,day,status); CREATE TABLE broadcast(id,channel,day,metadata,status,word_count);')

    def inventory(self, day, status='200'):
        self.db.execute('INSERT INTO inventory VALUES(?,?,?)', ('1TV', day.replace('-', ''), status))

    def broadcast(self, identifier, began, minutes=30, status='200', words=1000, program=''):
        start = dt.datetime.fromisoformat(began)
        metadata = dict(start_time=began, stop_time=str(start+dt.timedelta(minutes=minutes)), program=program)
        self.db.execute('INSERT INTO broadcast VALUES(?,?,?,?,?,?)', (identifier, '1TV', start.strftime('%Y%m%d'), json.dumps(metadata), status, words))

    def result(self, start, end=None):
        self.db.commit()
        result = load_coverage(self.path, start, end or start)
        return result, [r for r in result['daily'] if r['channel']=='1TV']

    def test_utc_boundary_requires_both_inventories(self):
        self.inventory('2023-12-31')
        self.inventory('2024-01-01')
        self.broadcast('night', '2023-12-31 21:00:00')
        result, daily = self.result('2024-01-01')
        self.assertEqual(result['broadcasts'][0]['day'], '2024-01-01')
        self.assertTrue(daily[0]['observed'])
        self.db.execute("UPDATE inventory SET status='404' WHERE day='20231231'")
        _, daily = self.result('2024-01-01')
        self.assertFalse(daily[0]['observed'])
        self.assertIn('previous_utc_inventory_unavailable', daily[0]['coverage_reasons'])

    def test_leap_day_month_boundary_and_empty_day(self):
        for date in ('2024-02-27','2024-02-28','2024-02-29','2024-03-01'):
            self.inventory(date)
        self.broadcast('leap', '2024-02-29 20:59:00')
        self.broadcast('march', '2024-02-29 21:00:00')
        _, daily = self.result('2024-02-28','2024-03-01')
        self.assertEqual([r['day'] for r in daily], ['2024-02-28','2024-02-29','2024-03-01'])
        self.assertEqual([r['canonical_listed_broadcasts'] for r in daily], [0,1,1])
        self.assertFalse(daily[0]['observed'])
        self.assertTrue(daily[2]['observed'])

    def test_same_start_longest_successful_and_repeat_preserved(self):
        self.inventory('2024-01-01'); self.inventory('2024-01-02')
        self.broadcast('unavailable', '2024-01-02 08:00:00', 90, '404')
        self.broadcast('short_named', '2024-01-02 08:00:00', 10, program='News')
        self.broadcast('long_z', '2024-01-02 08:00:00', 30)
        self.broadcast('long_a', '2024-01-02 08:00:00', 30)
        self.broadcast('repeat', '2024-01-02 08:29:00', 30)
        result, daily = self.result('2024-01-02')
        self.assertEqual(result['canonical_id_by_id']['short_named'], 'long_a')
        self.assertEqual(result['canonical_id_by_id']['repeat'], 'repeat')
        self.assertEqual(daily[0]['raw_listed_broadcasts'], 5)
        self.assertEqual(daily[0]['canonical_listed_broadcasts'], 2)
        self.assertEqual(daily[0]['overlapping_broadcasts'], 1)
        self.assertTrue(daily[0]['observed'])

    def test_low_word_long_success_is_not_observed(self):
        self.inventory('2024-01-01'); self.inventory('2024-01-02')
        self.broadcast('short_asr', '2024-01-02 08:00:00', words=6)
        _, daily = self.result('2024-01-02')
        self.assertFalse(daily[0]['observed'])
        self.assertEqual(daily[0]['canonical_available_broadcasts'], 1)
        self.assertIn('long_low_word_asr', daily[0]['coverage_reasons'])

    def test_overlap_context_before_requested_day_is_not_in_denominators(self):
        self.inventory('2024-01-01'); self.inventory('2024-01-02')
        self.broadcast('previous', '2024-01-01 20:50:00', 30)
        self.broadcast('current', '2024-01-01 21:00:00', 30)
        result, daily = self.result('2024-01-02')
        self.assertEqual(list(result['canonical_id_by_id']), ['current'])
        self.assertEqual(daily[0]['canonical_listed_broadcasts'], 1)
        self.assertEqual(result['broadcasts'][0]['overlap_previous_seconds'], 1200)


if __name__ == '__main__':
    unittest.main()
