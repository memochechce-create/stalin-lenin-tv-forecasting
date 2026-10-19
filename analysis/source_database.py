"""Read immutable collection tables with an optional successful repair overlay."""
from contextlib import contextmanager
from pathlib import Path
import sqlite3


@contextmanager
def open_source_database(base, supplemental=None):
    """Yield a Row connection exposing inventory/broadcast as merged TEMP views.

    Both attached files are read-only. Only successful supplemental records
    replace base records; failed repair attempts remain diagnostic data.
    Minimal fixture schemas are supported when both files share that schema.
    """
    connection = sqlite3.connect(':memory:', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('ATTACH DATABASE ? AS original', (Path(base).resolve().as_uri()+'?mode=ro',))
        overlay = supplemental is not None and Path(supplemental).exists()
        if overlay:
            connection.execute('ATTACH DATABASE ? AS supplemental', (Path(supplemental).resolve().as_uri()+'?mode=ro',))
        for table, keys in [('inventory', ('channel', 'day')), ('broadcast', ('id',))]:
            if overlay:
                match = ' AND '.join(f's.{key}=b.{key}' for key in keys)
                connection.execute(f'''CREATE TEMP VIEW {table} AS
                    SELECT b.* FROM original.{table} b WHERE NOT EXISTS
                      (SELECT 1 FROM supplemental.{table} s WHERE {match} AND s.status='200')
                    UNION ALL SELECT s.* FROM supplemental.{table} s WHERE s.status='200' ''')
            else:
                connection.execute(f'CREATE TEMP VIEW {table} AS SELECT * FROM original.{table}')
        yield connection
    finally:
        connection.close()
