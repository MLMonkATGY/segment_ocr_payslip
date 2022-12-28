docker exec -i job_oracle_postgresdb_1 /bin/bash -c "PGPASSWORD=alex_prometheus pg_dump --username alex_prometheus job_etl" > ./job_oracle_db_dump.sql
