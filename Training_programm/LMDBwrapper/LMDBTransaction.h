#ifndef LMDBTRANSACTION_H
#define LMDBTRANSACTION_H

#include <lmdb.h>
#include <vector>
#include <string>
#include <assert.h>

class LMDBTransaction
{
public:
    explicit LMDBTransaction(MDB_env* mdb_env) : mdb_env_(mdb_env) { }
    void Put(const std::string& key, const std::string& value);
    void Commit();

    LMDBTransaction(LMDBTransaction&) = delete;
    void operator=(LMDBTransaction) = delete;

private:
    MDB_env* mdb_env_;
    std::vector<std::string> keys, values;

    void DoubleMapSize();
};

#endif // LMDBTRANSACTION_H
