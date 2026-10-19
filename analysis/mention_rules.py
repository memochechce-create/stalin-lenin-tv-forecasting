"""Versioned, conservative context rules; labels are not validated ground truth."""
import re

VERSION = 'context-rules-2'
SEP = r'[\s.,:;«»"“”„\-–—]*(?:имени\s+|им\.\s*)?'
ADDRESS = re.compile(r'\b(?:улиц\w*|ул\.|проспект\w*|просп\.|площад\w*|пл\.|переул\w*|бульвар\w*|шоссе|набережн\w*|станци\w*|метро|пос[её]л\w*|село|села|город\w*|район\w*|парк\w*|сквер\w*)'+SEP+r'$', re.I)
OBJECT = re.compile(r'\b(?:ледокол\w*|теплоход\w*|пароход\w*|корабл\w*|библиотек\w*|школ\w*|завод\w*|автозавод\w*|университет\w*|институт\w*|стадион\w*|совхоз\w*|колхоз\w*)'+SEP+r'$', re.I)
COMMEMORATION = re.compile(r'\b(?:памятник\w*|бюст\w*|мавзоле\w*|орден\w*|преми\w*|портрет\w*|монумент\w*)'+SEP+r'$', re.I)
NAME = {'lenin': re.compile(r'\b(?:владимир\w*\s+(?:ильич\w*\s+)?|в\s*\.\s*(?:и\s*\.\s*)?)$', re.I),
        'stalin': re.compile(r'\b(?:иосиф\w*\s+(?:виссарионович\w*\s+)?|и\s*\.\s*(?:в\s*\.\s*)?)$', re.I)}
OTHER_NAME = re.compile(r'\b(?:елен\w*|лен(?:а|ы|е|у|ой)|васили\w*|светлан\w*|юли\w*|евгени\w*|александр\w*|андре\w*|никола\w*|дмитри\w*|леонид\w*)\s+$', re.I)


def classify(m):
    prefix = m['context'][:m['context_offset']]
    suffix = m['context'][m['context_offset']+len(m['matched']):]
    person = m['person']
    # Collection of local city names is evidence of an ASR ambiguity, not a correction.
    nearby = (prefix[-100:]+' '+suffix[:100]).lower()
    if person == 'stalin' and ('вильнюс' in nearby or 'рига' in nearby or
            ('боеприпас' in nearby and ('киев' in nearby or 'евро' in nearby))):
        return 'uncertain_asr', 'possible_tallinn_asr'
    full_name=NAME[person].search(prefix)
    object_prefix=prefix[:full_name.start()] if full_name else prefix
    if ADDRESS.search(object_prefix):
        return 'named_object', 'address_or_station'
    if OBJECT.search(object_prefix):
        return 'named_object', 'named_institution_or_vehicle'
    if full_name:
        return 'person_likely', 'historical_full_name'
    if person=='lenin' and re.search(r'\bникола\w*\s+$',prefix,re.I):
        if re.search(r'ульянов|псевдоним|имена|вожд|философск|политическ',nearby,re.I):
            return 'person_likely','historical_pseudonym_nikolai'
        return 'uncertain_reference','nikolai_lenin_alias_or_other_person'
    if OTHER_NAME.search(prefix):
        if re.search(r'\b(?:елен\w*|лен(?:а|ы|е|у|ой)|васили\w*|светлан\w*)\s+$',prefix,re.I):
            return 'other_person', 'different_given_name'
        return 'uncertain_reference', 'possible_other_given_name_or_asr_attachment'
    if re.search(r'освобо\w*\s+$',prefix,re.I) and re.search(r'всу|армия|бойц|фронт|войск|днепропетров',nearby,re.I):
        return 'named_object','military_locality_context'
    if re.search(r'ледокол|причал|теплоход',suffix[:160],re.I):
        return 'uncertain_reference','possible_vehicle_in_following_context'
    if re.search(r'\bот\s+имени\s*$',prefix,re.I):
        return 'person_likely','on_behalf_of_person'
    if re.search(r'преми\w*\s*$',prefix,re.I) and m['matched'].lower().endswith('у'):
        return 'person_likely','award_recipient'
    if m['matched'].lower() in ('сталин','ленин') and re.search(r'^\s+(?:призвал|сказал|выступил|пишет|писал)\b',suffix,re.I):
        return 'person_likely','person_subject_with_speech_verb'
    if COMMEMORATION.search(prefix):
        return 'uncertain_reference', 'commemoration_or_personal_possession'
    if re.search(r'\b(?:имени|им\.)\s*$', prefix, re.I):
        return 'uncertain_reference', 'unspecified_named_object'
    if m.get('nonstandard_asr_form'):
        return 'uncertain_asr', 'nonstandard_instrumental_asr'
    return 'person_likely', 'surname_context_default_unvalidated'


def normalize_tokens(text):
    return [(m.group().lower().replace('ё','е'),m.start(),m.end()) for m in re.finditer(r'\w+',text)]
