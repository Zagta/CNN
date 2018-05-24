#ifndef LMDBCURSOR_H
#define LMDBCURSOR_H

#include <lmdb.h>
#include <vector>
#include <string>
#include <assert.h>

class LMDBCursor
{
public:
    LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor) : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false)
    {
        SeekToFirst();
    }

    ~LMDBCursor()
    {
        mdb_cursor_close(mdb_cursor_);
        mdb_txn_abort(mdb_txn_);
    }

    MDB_val mdb_key_, mdb_value_;

    void SeekToFirst() { Seek(MDB_FIRST); }
    void SeekToLast() { Seek(MDB_LAST); }
    void Next() { Seek(MDB_NEXT); }
    void GetByKey(std::string tmp)
    {
        mdb_key_.mv_size = tmp.size();
        mdb_key_.mv_data = const_cast<char*>(tmp.data());
        Seek(MDB_SET_KEY);
    }

    std::string key()
    {
        return std::string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
    }

    std::string value()
    {
        return std::string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
    }

    bool valid() { return valid_; }

    LMDBCursor(LMDBCursor&) = delete;
    void operator=(LMDBCursor) = delete;

private:
    void Seek(MDB_cursor_op op);

    MDB_txn* mdb_txn_;
    MDB_cursor* mdb_cursor_;
    bool valid_;
};

#endif // LMDBCURSOR_H
