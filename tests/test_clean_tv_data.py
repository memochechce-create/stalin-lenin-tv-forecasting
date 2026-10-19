import unittest
from analysis.clean_tv_data import calendar, duplicate_pairs
from analysis.mention_rules import classify


def mention(prefix, name='Ленина', person='lenin', suffix=''):
    return dict(person=person,matched=name,context=prefix+name+suffix,context_offset=len(prefix))


class ContextTests(unittest.TestCase):
    def test_address_and_person_are_different(self):
        self.assertEqual(classify(mention('гололёд на площади '))[0], 'named_object')
        self.assertEqual(classify(mention('слова Владимира Ильича '))[0], 'person_likely')
        self.assertEqual(classify(mention('обсуждаем труды '))[0], 'person_likely')
        self.assertEqual(classify(mention('на улице имени '))[0], 'named_object')
        self.assertEqual(classify(mention('на улице имени Владимира Ильича '))[0], 'named_object')
        self.assertEqual(classify(mention('в парке '))[0], 'named_object')

    def test_possession_is_not_automatically_discarded(self):
        self.assertEqual(classify(mention('Ордена ', 'Сталина','stalin',', его мундир'))[0], 'uncertain_reference')
        self.assertEqual(classify(mention('памятник '))[0], 'uncertain_reference')

    def test_other_person_and_asr(self):
        self.assertEqual(classify(mention('Елена ','Ленина'))[0], 'other_person')
        self.assertEqual(classify(mention('Рига, Вильнюс, ','Сталин','stalin'))[0], 'uncertain_asr')
        self.assertEqual(classify(mention('Ленин ','Сталин','stalin'))[0], 'person_likely')
        self.assertEqual(classify(mention('от имени ','Сталина','stalin'))[0], 'person_likely')
        self.assertEqual(classify(mention('с трибуны мавзолея ','Сталин','stalin',' призвал бойцов'))[0], 'person_likely')
        self.assertEqual(classify(mention('Ульянов, Николай ','Ленин'))[0], 'person_likely')

    def test_calendar_crosses_year_and_separates_may8_may9(self):
        self.assertEqual(calendar('2024-05-08')['is_may8'],1)
        self.assertEqual(calendar('2024-05-08')['is_may9'],0)
        self.assertEqual(calendar('2024-11-07')['is_nov7'],1)

    def test_identical_rebroadcast_outside_overlap_is_preserved(self):
        a=[dict(archive_start='2024-01-01T08:00:00+00:00',archive_stop='2024-01-01T08:30:00+00:00')]
        b=[dict(archive_start='2024-01-01T09:00:00+00:00',archive_stop='2024-01-01T09:30:00+00:00')]
        self.assertEqual(duplicate_pairs(a,b,{},{}),[])

    def test_dedup_requires_text_evidence_and_compatible_timing(self):
        text='один два три четыре пять шесть семь восемь девять десять Ленин затем речь продолжается и ещё несколько слов завершают длинную фразу сегодня'
        offset=text.index('Ленин')
        old=dict(person='lenin',offset=offset,start_seconds=1755,archive_start='2024-01-01T01:00:00+00:00',archive_stop='2024-01-01T01:30:00+00:00')
        new=dict(person='lenin',offset=offset,start_seconds=15,archive_start='2024-01-01T01:29:00+00:00',archive_stop='2024-01-01T01:59:00+00:00')
        source=dict(head=text,tail=text,chars=len(text))
        self.assertEqual(len(duplicate_pairs([old],[new],source,source)),1)
        new['start_seconds']=45
        self.assertEqual(duplicate_pairs([old],[new],source,source),[])
        new['start_seconds']=None; old['start_seconds']=None
        self.assertEqual(len(duplicate_pairs([old],[new],source,source)),1)


if __name__=='__main__':
    unittest.main()
