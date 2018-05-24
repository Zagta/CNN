#ifndef LMDB_H
#define LMDB_H

#include "LMDBCursor.h"
#include "LMDBTransaction.h"
#include "assert.h"
#include <sys/stat.h>

enum Mode { READ, WRITE, NEW };

class LMDB
{
public:
    LMDB() : mdb_env_(NULL) { }
    ~LMDB() { Close(); }
    void Open(const std::string& source, Mode mode);
    void Close();

    LMDBCursor* NewCursor();
    LMDBTransaction* NewTransaction();

    LMDB(LMDB&) = delete;
    void operator=(LMDB) = delete;

private:
    MDB_env* mdb_env_;
    MDB_dbi mdb_dbi_;
};

#endif // LMDB_H
