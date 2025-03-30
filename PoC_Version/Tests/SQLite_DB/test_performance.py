# tests/test_performance.py


#
####################################################################################################
# Test Status:
# working as of 2024-10-01 - This test checks the performance of the search_media_database function
# by populating the database with a specified number of records and measuring the time taken to search
# 10, 100, 1k, 10k records. The expected search time never exceeds 2s for 10k records.
# This can be/should be tuned down more.

# @pytest.fixture(scope="module")
# def db():
#     test_db_name = 'test_performance.db'
#     database = Database(test_db_name)
#     create_tables(database)
#
#     # Ensure the database is empty before starting tests
#     with database.get_connection() as conn:
#         conn.execute("DELETE FROM Media")
#         conn.execute("DELETE FROM MediaKeywords")
#         conn.execute("DELETE FROM Keywords")
#
#     yield database
#
#     # Ensure all connections are closed properly
#     database.close_connection()
#
#     # Cleanup the test database file after all tests
#     db_path = Utils.get_database_path(test_db_name)
#     if os.path.exists(db_path):
#         os.remove(db_path)

#@pytest.mark.parametrize("num_records", [10, 100, 1000, 10000])
# def test_search_performance(db, num_records):
#     # Get a connection from the database
#     with db.get_connection() as conn:
#         # Populate the database with test records
#         for i in range(num_records):
#             url = f'https://example.com/perf_test_{i}'
#             info_dict = {'title': f'Performance Test Video {i}', 'uploader': 'Test Uploader'}
#             # Pass the connection explicitly
#             add_media_to_database(url, info_dict, [], 'Test summary', ['performance'],
#                                   'Test prompt', 'whisper-1', overwrite=True, db=db)
#
#         # Perform the search
#         start_time = time.time()
#         # Pass the connection explicitly to the search function
#         results = search_media_database('Performance Test', connection=conn)
#         end_time = time.time()
#
#         # Check results
#         assert len(results) == num_records
#
#         # Adjust performance expectations based on number of records
#         if num_records <= 10:
#             max_time = 0.1
#         elif num_records <= 100:
#             max_time = 0.5
#         else:
#             max_time = 2.0
#
#         search_time = end_time - start_time
#         print(f"Search time for {num_records} records: {search_time:.4f} seconds")
#         assert search_time < max_time, f"Search took {search_time:.4f} seconds, which is more than the expected {max_time} seconds for {num_records} records"
#
#         # Clean up the database after the test
#         conn.execute("DELETE FROM Media")
#         conn.execute("DELETE FROM MediaKeywords")
#         conn.execute("DELETE FROM Keywords")
#
# End of File
####################################################################################################
