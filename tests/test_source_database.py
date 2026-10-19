import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from analysis.source_database import open_source_database


class OverlayTests(unittest.TestCase):
    def test_only_successful_repairs_override_and_sources_are_read_only(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root=Path(temp)
            for filename in ('base.sqlite','repair.sqlite'):
                with closing(sqlite3.connect(root/filename)) as c, c:
                    c.executescript('CREATE TABLE inventory(channel,day,status); CREATE TABLE broadcast(id,status);')
            with closing(sqlite3.connect(root/'base.sqlite')) as c, c:
                c.executemany('INSERT INTO broadcast VALUES(?,?)',[('same','200'),('missing','404')])
            with closing(sqlite3.connect(root/'repair.sqlite')) as c, c:
                c.executemany('INSERT INTO broadcast VALUES(?,?)',[('same','404'),('missing','200'),('new','200')])
            with open_source_database(root/'base.sqlite',root/'repair.sqlite') as c:
                self.assertEqual(dict(c.execute('SELECT id,status FROM broadcast')),
                                 {'same':'200','missing':'200','new':'200'})
                with self.assertRaises(sqlite3.OperationalError):
                    c.execute("UPDATE original.broadcast SET status='bad'")


if __name__=='__main__': unittest.main()
