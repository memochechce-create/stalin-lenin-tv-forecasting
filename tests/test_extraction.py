import unittest
from analysis.extract_tv_mentions import PATTERN, parse_source


class ExtractionTests(unittest.TestCase):
    def test_word_boundaries_and_nonstandard_asr_forms(self):
        text='Ленин Лениным Ленином Сталиным Сталином Ленинград сталинизм xЛенин Ленин123 _Ленин'
        self.assertEqual([m.group() for m in PATTERN.finditer(text)],['Ленин','Лениным','Ленином','Сталиным','Сталином'])

    def test_invalid_srt_timing_keeps_speech(self):
        raw=b'1\n00:00:05,000 --> 00:00:04,000\nSpeech\n\n2\n00:00:07,100 --> 00:00:08,300\nMore words\n'
        text,cues=parse_source(raw,'srt')
        self.assertEqual(text,'Speech\nMore words')
        self.assertIsNone(cues[0][2])
        self.assertEqual(cues[1],(7,17,7.1,8.3))


if __name__=='__main__': unittest.main()
