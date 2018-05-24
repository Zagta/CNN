#include "LMDBTransaction.h"

void LMDBTransaction::Put(const std::string& key, const std::string& value)
{
    keys.push_back(key);
    values.push_back(value);
}

void LMDBTransaction::Commit()
{
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    // Initialize MDB variables
    int mdb_txn_status = mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn);
    assert(mdb_txn_status == MDB_SUCCESS && "mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn) failed");
    int mdb_dbi_open_status = mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi);
    assert(mdb_dbi_open_status == MDB_SUCCESS && "mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi) failed");

    for (int i = 0; i < keys.size(); i++)
    {
        mdb_key.mv_size = keys[i].size();
        mdb_key.mv_data = const_cast<char*>(keys[i].data());
        mdb_data.mv_size = values[i].size();
        mdb_data.mv_data = const_cast<char*>(values[i].data());

        // Add data to the transaction
        int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
        if (put_rc == MDB_MAP_FULL)
        {
            // Out of memory - double the map size and retry
            mdb_txn_abort(mdb_txn);
            mdb_dbi_close(mdb_env_, mdb_dbi);
            DoubleMapSize();
            Commit();
            return;
        }

        // May have failed for some other reason
        assert(put_rc == MDB_SUCCESS && "mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0) failed");
    }

    // Commit the transaction
    int commit_rc = mdb_txn_commit(mdb_txn);
    if (commit_rc == MDB_MAP_FULL)
    {
        // Out of memory - double the map size and retry
        mdb_dbi_close(mdb_env_, mdb_dbi);
        DoubleMapSize();
        Commit();
        return;
    }
    // May have failed for some other reason
    assert(commit_rc == MDB_SUCCESS && "mdb_txn_commit(mdb_txn) failed");

    // Cleanup after successful commit
    mdb_dbi_close(mdb_env_, mdb_dbi);
    keys.clear();
    values.clear();
}

void LMDBTransaction::DoubleMapSize()
{
    struct MDB_envinfo current_info;
    int mdb_env_info_status = mdb_env_info(mdb_env_, &current_info);
    assert(mdb_env_info_status == MDB_SUCCESS && "mdb_env_info(mdb_env_, &current_info) failed");
    size_t new_size = current_info.me_mapsize * 2;

    int mdb_env_set_mapsize_staus = mdb_env_set_mapsize(mdb_env_, new_size);
    assert(mdb_env_set_mapsize_staus == MDB_SUCCESS && "mdb_env_set_mapsize(mdb_env_, new_size) failed");
}
