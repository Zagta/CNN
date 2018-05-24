#include "LMDB.h"

void LMDB::Open(const std::string& source, Mode mode)
{
    int mdb_status = mdb_env_create(&mdb_env_);
    assert(mdb_status == MDB_SUCCESS && "mdb_env_create(&mdb_env_) failed");

    if (mode == NEW)
    {
        int mkdir_status = mkdir(source.c_str(), 0744);
        assert(mkdir_status == 0 && "LMDB mkdir failed");
    }

    int flags = 0;

    if (mode == READ)
    {
        flags = MDB_RDONLY | MDB_NOTLS;
    }

    int rc = mdb_env_open(mdb_env_, source.c_str(), flags, 0664);
    assert(rc == MDB_SUCCESS && "mdb_strerror(rc) failed");
}

LMDBCursor* LMDB::NewCursor()
{
    MDB_txn* mdb_txn;
    MDB_cursor* mdb_cursor;
    int mdb_txn_begin_status = mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn);
    assert(mdb_txn_begin_status == MDB_SUCCESS && "mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn) failed");
    int mdb_dbi_open_status = mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_);
    assert(mdb_dbi_open_status == MDB_SUCCESS && "mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_) failed");
    int mdb_cursor_open_status = mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor);
    assert(mdb_cursor_open_status == MDB_SUCCESS && "mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor) failed");
    return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::NewTransaction()
{
    return new LMDBTransaction(mdb_env_);
}

void LMDB::Close()
{
    if (mdb_env_ != NULL)
    {
        mdb_dbi_close(mdb_env_, mdb_dbi_);
        mdb_env_close(mdb_env_);
        mdb_env_ = NULL;
    }
}
