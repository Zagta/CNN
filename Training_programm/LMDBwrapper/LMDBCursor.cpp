#include "LMDBCursor.h"

void LMDBCursor::Seek(MDB_cursor_op op)
{
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND)
    {
        valid_ = false;
    }
    else
    {
        assert(mdb_status == MDB_SUCCESS && "mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op) failed");
        valid_ = true;
    }
}
