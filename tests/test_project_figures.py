import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
import project_figures as pf

class PlottingTests(unittest.TestCase):
    def test_joint_sum_not_sum_of_quantiles(self):
        # Two mutually exclusive channel spikes cancel aggregate uncertainty.
        x=np.zeros((2,1,4),dtype=int);x[0,0,0]=10;x[1,0,1]=10
        out=pf.aggregate_daily_paths({key:x for key in pf.rv.KEYS})
        for v in out.values():
            np.testing.assert_array_equal(v,[[10],[10]])
            self.assertEqual(np.quantile(v,.9),10)
        self.assertEqual(np.quantile(x,.9,axis=0).sum(),18)

    def test_incomplete_channel_day_is_gap(self):
        rows=[]
        for day in pd.date_range('2026-10-01',periods=2):
            for c,ch in enumerate(pf.m.CHANNELS):
                rows.append(dict(date=day,channel=ch,observed=not(day.day==2 and c==0),**{key:1 for key in pf.rv.KEYS}))
        out=pf.national_daily(pd.DataFrame(rows))
        self.assertEqual(len(out),1)
        self.assertTrue((out.iloc[0]==4).all())

    def test_wrong_channel_shape_rejected(self):
        with self.assertRaises(ValueError):
            pf.aggregate_daily_paths({key:np.zeros((2,3,3)) for key in pf.rv.KEYS})

    def test_save_text_normalizes_long_dashes(self):
        fig,ax=pf.plt.subplots();ax.set_title('A\u2014B\u2013C\u2015D');ax.text(0,0,'one\u2012two')
        pf.clean_figure_text(fig)
        for obj in fig.findobj(match=pf.Text):
            self.assertFalse(set(obj.get_text()) & set(map(chr,range(0x2012,0x2016))))
        pf.plt.close(fig)

if __name__=='__main__':unittest.main()
