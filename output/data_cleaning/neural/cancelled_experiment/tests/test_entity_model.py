import unittest
from analysis.entity_model import model_text,input_key,decision_label


class EntityTests(unittest.TestCase):
    def test_only_the_selected_occurrence_is_marked(self):
        text='Ленин выступил. На площади Ленина прошёл митинг.'
        pos=text.index('Ленина')
        m=dict(context=text,context_offset=pos,matched='Ленина',person='lenin')
        out=model_text(m)
        self.assertIn('Ленин выступил.',out)
        self.assertIn('площади [MENTION] Ленина [/MENTION]',out)
        self.assertEqual(out.count('[MENTION] Ленина [/MENTION]'),1)

    def test_invalid_offsets_fail_instead_of_classifying_a_different_name(self):
        with self.assertRaises(ValueError):
            model_text(dict(context='Сталин',context_offset=0,matched='Ленин',person='lenin'))

    def test_cache_key_includes_target_identity(self):
        a=dict(context='Сталин',context_offset=0,matched='Сталин',person='stalin')
        b=dict(a,person='lenin')
        self.assertNotEqual(input_key(a,model_sha='fixture'),input_key(b,model_sha='fixture'))

    def test_low_confidence_is_not_accepted_by_default(self):
        for category in ('historical_person','non_target_reference','commemoration'):
            self.assertEqual(decision_label(dict(category=category,score=.4,margin=.05),.7,.15)[0],'uncertain_reference')
        self.assertEqual(decision_label(dict(category='historical_person',score=.95,margin=.8),.7,.15)[0],'person_likely')
        self.assertEqual(decision_label(dict(category='non_target_reference',score=.95,margin=.8),.7,.15)[0],'not_target_person')


if __name__=='__main__': unittest.main()
