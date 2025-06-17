# test_performance_benchmarks.py
# Description: Performance benchmarks to validate RAG optimizations
#
# Imports
import pytest
import json
import time
import statistics
import psutil
import os
from unittest.mock import patch
from pathlib import Path
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RAG.RAG_Unified_Library_v2 import (
    fetch_relevant_ids_by_keywords,
    DatabaseType
)
from tldw_Server_API.app.core.config import RAG_SEARCH_CONFIG
#
#######################################################################################################################
#
# Test Fixtures and Utilities

@pytest.fixture
def client_id():
    return "benchmark_client"

@pytest.fixture
def mem_db(client_id):
    """Creates an in-memory CharactersRAGDB for benchmarking."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()

def create_large_character_dataset(db, num_characters=1000):
    """Create a large dataset of character cards for performance testing."""
    character_ids = []
    
    # Pre-defined tag pools for variety
    tag_pools = [
        ["fantasy", "magic", "wizard", "spell", "arcane"],
        ["sci-fi", "robot", "android", "space", "technology"],
        ["adventure", "explorer", "quest", "treasure", "journey"],
        ["mystery", "detective", "crime", "investigation", "clue"],
        ["romance", "love", "passion", "heart", "emotion"],
        ["horror", "scary", "dark", "ghost", "supernatural"],
        ["comedy", "funny", "humor", "laugh", "joke"],
        ["drama", "serious", "intense", "conflict", "emotion"],
        ["action", "fight", "battle", "warrior", "combat"],
        ["slice-of-life", "everyday", "normal", "casual", "simple"]
    ]
    
    for i in range(num_characters):
        # Select tags from different pools for variety
        tag_pool_index = i % len(tag_pools)
        num_tags = min(3 + (i % 4), len(tag_pools[tag_pool_index]))  # 3-6 tags
        char_tags = tag_pools[tag_pool_index][:num_tags]
        
        # Add some common tags for search testing
        if i % 10 == 0:
            char_tags.append("featured")
        if i % 5 == 0:
            char_tags.append("popular")
        if i % 3 == 0:
            char_tags.append("recommended")
        
        char_data = {
            "name": f"Character {i:04d}",
            "description": f"Description for character {i}. " * (5 + i % 10),  # Variable length
            "personality": f"Personality traits for character {i}. " * (3 + i % 5),
            "scenario": f"Scenario setting for character {i}. " * (2 + i % 3),
            "tags": json.dumps(char_tags),
            "first_message": f"Hello! I'm character {i}. Nice to meet you!"
        }
        
        char_id = db.add_character_card(char_data)
        assert char_id is not None
        character_ids.append(char_id)
    
    return character_ids

def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def benchmark_function(func, *args, num_runs=5, **kwargs):
    """Benchmark a function and return timing statistics."""
    times = []
    memory_before = measure_memory_usage()
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    memory_after = measure_memory_usage()
    
    return {
        'times': times,
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
        'memory_delta_mb': memory_after - memory_before,
        'result': result  # Return last result for validation
    }

#
# Benchmark Test Classes

class TestCharacterCardTagSearchPerformance:
    """Performance benchmarks for character card tag search optimization."""
    
    @pytest.mark.benchmark
    def test_tag_search_small_dataset_performance(self, mem_db):
        """Benchmark tag search performance with small dataset (100 characters)."""
        num_characters = 100
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Benchmark single tag search
        single_tag_stats = benchmark_function(
            fetch_relevant_ids_by_keywords,
            None,  # media_db
            mem_db,  # char_rag_db
            DatabaseType.CHARACTER_CARDS,
            ["fantasy"],
            num_runs=10
        )
        
        # Benchmark multiple tag search
        multi_tag_stats = benchmark_function(
            fetch_relevant_ids_by_keywords,
            None,
            mem_db,
            DatabaseType.CHARACTER_CARDS,
            ["fantasy", "sci-fi", "adventure"],
            num_runs=10
        )
        
        # Performance assertions for small dataset
        assert single_tag_stats['mean_time'] < 0.1, f"Single tag search took {single_tag_stats['mean_time']:.3f}s, expected < 0.1s"
        assert multi_tag_stats['mean_time'] < 0.15, f"Multi tag search took {multi_tag_stats['mean_time']:.3f}s, expected < 0.15s"
        assert single_tag_stats['memory_delta_mb'] < 10, f"Memory usage increased by {single_tag_stats['memory_delta_mb']:.2f}MB, expected < 10MB"
        
        # Log performance metrics
        print(f"\nSmall Dataset ({num_characters} characters) Performance:")
        print(f"Single tag search: {single_tag_stats['mean_time']:.4f}s ± {single_tag_stats['std_dev']:.4f}s")
        print(f"Multi tag search: {multi_tag_stats['mean_time']:.4f}s ± {multi_tag_stats['std_dev']:.4f}s")
        print(f"Memory delta: {single_tag_stats['memory_delta_mb']:.2f}MB")
    
    @pytest.mark.benchmark  
    def test_tag_search_medium_dataset_performance(self, mem_db):
        """Benchmark tag search performance with medium dataset (1000 characters)."""
        num_characters = 1000
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Benchmark various search scenarios
        scenarios = [
            (["fantasy"], "Single common tag"),
            (["featured"], "Single rare tag"),
            (["fantasy", "sci-fi"], "Two common tags"),
            (["featured", "popular"], "Two rare tags"),
            (["fantasy", "sci-fi", "adventure", "mystery"], "Four tags"),
            (["nonexistent"], "Nonexistent tag")
        ]
        
        results = {}
        for tags, description in scenarios:
            stats = benchmark_function(
                fetch_relevant_ids_by_keywords,
                None,
                mem_db, 
                DatabaseType.CHARACTER_CARDS,
                tags,
                num_runs=5
            )
            results[description] = stats
            
            # Performance assertion - should complete in reasonable time
            assert stats['mean_time'] < 0.5, f"{description} took {stats['mean_time']:.3f}s, expected < 0.5s"
        
        # Log detailed performance metrics
        print(f"\nMedium Dataset ({num_characters} characters) Performance:")
        for description, stats in results.items():
            print(f"{description:20}: {stats['mean_time']:.4f}s ± {stats['std_dev']:.4f}s "
                  f"(found {len(stats['result'])} results)")
    
    @pytest.mark.benchmark
    def test_tag_search_large_dataset_performance(self, mem_db):
        """Benchmark tag search performance with large dataset (5000 characters)."""
        num_characters = 5000
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Test scalability - performance should not degrade linearly
        stats = benchmark_function(
            fetch_relevant_ids_by_keywords,
            None,
            mem_db,
            DatabaseType.CHARACTER_CARDS,
            ["fantasy"],
            num_runs=3  # Fewer runs for large dataset
        )
        
        # Even with 5000 characters, should complete quickly due to database optimization
        assert stats['mean_time'] < 1.0, f"Large dataset search took {stats['mean_time']:.3f}s, expected < 1.0s"
        assert stats['memory_delta_mb'] < 50, f"Memory usage increased by {stats['memory_delta_mb']:.2f}MB, expected < 50MB"
        
        print(f"\nLarge Dataset ({num_characters} characters) Performance:")
        print(f"Tag search: {stats['mean_time']:.4f}s ± {stats['std_dev']:.4f}s")
        print(f"Memory delta: {stats['memory_delta_mb']:.2f}MB")
        print(f"Results found: {len(stats['result'])}")
    
    @pytest.mark.benchmark
    def test_concurrent_search_performance(self, mem_db):
        """Benchmark concurrent tag search performance."""
        import threading
        import queue
        
        # Create dataset
        num_characters = 500
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Define concurrent search workload
        search_tasks = [
            ["fantasy"],
            ["sci-fi"],
            ["adventure"],
            ["mystery"],
            ["featured"],
            ["popular"],
            ["fantasy", "magic"],
            ["sci-fi", "robot"],
            ["adventure", "quest"],
            ["mystery", "detective"]
        ]
        
        results_queue = queue.Queue()
        start_time = time.perf_counter()
        
        def search_worker(tags):
            worker_start = time.perf_counter()
            result_ids = fetch_relevant_ids_by_keywords(
                None, mem_db, DatabaseType.CHARACTER_CARDS, tags
            )
            worker_end = time.perf_counter()
            results_queue.put({
                'tags': tags,
                'time': worker_end - worker_start,
                'results': len(result_ids)
            })
        
        # Start concurrent searches
        threads = []
        for tags in search_tasks:
            thread = threading.Thread(target=search_worker, args=(tags,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        total_time = time.perf_counter() - start_time
        
        # Collect results
        worker_results = []
        while not results_queue.empty():
            worker_results.append(results_queue.get())
        
        assert len(worker_results) == len(search_tasks), "Not all searches completed"
        
        # Concurrent execution should be much faster than sequential
        max_worker_time = max(result['time'] for result in worker_results)
        assert total_time < max_worker_time * 2, f"Concurrent execution not efficient: {total_time:.3f}s vs max worker {max_worker_time:.3f}s"
        
        print(f"\nConcurrent Search Performance ({len(search_tasks)} searches):")
        print(f"Total time: {total_time:.4f}s")
        print(f"Max worker time: {max_worker_time:.4f}s")
        print(f"Average worker time: {sum(r['time'] for r in worker_results) / len(worker_results):.4f}s")
    
    @pytest.mark.benchmark
    def test_json_vs_fallback_performance_comparison(self, mem_db):
        """Compare performance of JSON method vs fallback method."""
        num_characters = 1000
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Test if JSON functions are available
        json_supported = mem_db._check_json_support()
        
        if json_supported:
            # Benchmark JSON method
            json_stats = benchmark_function(
                mem_db._search_cards_by_tags_json,
                ["fantasy"],
                10,  # limit
                num_runs=5
            )
            
            # Benchmark fallback method
            fallback_stats = benchmark_function(
                mem_db._search_cards_by_tags_fallback,
                ["fantasy"],
                10,  # limit
                num_runs=5
            )
            
            # JSON method should be faster (or at least not significantly slower)
            json_time = json_stats['mean_time']
            fallback_time = fallback_stats['mean_time']
            
            print(f"\nJSON vs Fallback Performance Comparison:")
            print(f"JSON method: {json_time:.4f}s ± {json_stats['std_dev']:.4f}s")
            print(f"Fallback method: {fallback_time:.4f}s ± {fallback_stats['std_dev']:.4f}s")
            print(f"JSON speedup: {fallback_time/json_time:.2f}x")
            
            # Verify results are consistent
            assert len(json_stats['result']) == len(fallback_stats['result']), "JSON and fallback methods returned different result counts"
        else:
            print("\nSQLite JSON functions not available - skipping comparison")


class TestMemoryEfficiencyBenchmarks:
    """Benchmarks for memory efficiency improvements."""
    
    @pytest.mark.benchmark
    def test_memory_usage_scaling(self, mem_db):
        """Test that memory usage doesn't scale linearly with dataset size."""
        dataset_sizes = [100, 500, 1000, 2000]
        memory_measurements = []
        
        for size in dataset_sizes:
            # Create fresh database for each test
            fresh_db = CharactersRAGDB(":memory:", "memory_test")
            
            try:
                # Measure baseline memory
                baseline_memory = measure_memory_usage()
                
                # Create dataset
                character_ids = create_large_character_dataset(fresh_db, size)
                
                # Measure memory after data creation
                data_memory = measure_memory_usage()
                
                # Perform multiple searches
                for _ in range(10):
                    result_ids = fetch_relevant_ids_by_keywords(
                        None, fresh_db, DatabaseType.CHARACTER_CARDS, ["fantasy"]
                    )
                
                # Measure memory after searches
                search_memory = measure_memory_usage()
                
                memory_measurements.append({
                    'size': size,
                    'data_overhead': data_memory - baseline_memory,
                    'search_overhead': search_memory - data_memory,
                    'total_overhead': search_memory - baseline_memory
                })
                
            finally:
                fresh_db.close_connection()
        
        # Analyze memory scaling
        print(f"\nMemory Usage Scaling Analysis:")
        for measurement in memory_measurements:
            print(f"Dataset {measurement['size']:4d}: "
                  f"Data {measurement['data_overhead']:6.2f}MB, "
                  f"Search {measurement['search_overhead']:6.2f}MB, "
                  f"Total {measurement['total_overhead']:6.2f}MB")
        
        # Memory usage should not scale linearly with dataset size for searches
        # (data storage will scale linearly, but search overhead should be minimal)
        largest_search_overhead = memory_measurements[-1]['search_overhead']
        assert largest_search_overhead < 20, f"Search memory overhead too high: {largest_search_overhead:.2f}MB"
    
    @pytest.mark.benchmark
    def test_memory_leak_detection(self, mem_db):
        """Test for memory leaks in repeated operations."""
        # Create dataset
        character_ids = create_large_character_dataset(mem_db, 500)
        
        # Baseline memory measurement
        baseline_memory = measure_memory_usage()
        
        # Perform many search operations
        for i in range(100):
            search_tags = [f"tag{i % 10}"] if i % 10 != 0 else ["fantasy", "sci-fi"]
            result_ids = fetch_relevant_ids_by_keywords(
                None, mem_db, DatabaseType.CHARACTER_CARDS, search_tags
            )
            
            # Periodically check memory
            if i % 20 == 19:
                current_memory = measure_memory_usage()
                memory_increase = current_memory - baseline_memory
                
                # Memory increase should be minimal (< 5MB) even after many operations
                assert memory_increase < 5, f"Potential memory leak detected: {memory_increase:.2f}MB increase after {i+1} operations"
        
        final_memory = measure_memory_usage()
        total_increase = final_memory - baseline_memory
        
        print(f"\nMemory Leak Test (100 operations):")
        print(f"Memory increase: {total_increase:.2f}MB")
        
        assert total_increase < 10, f"Memory leak detected: {total_increase:.2f}MB increase"


class TestScalabilityBenchmarks:
    """Benchmarks for scalability with different dataset sizes."""
    
    @pytest.mark.benchmark
    def test_search_time_scaling(self, mem_db):
        """Test that search time doesn't scale linearly with dataset size."""
        dataset_sizes = [100, 500, 1000, 2000, 5000]
        scaling_results = []
        
        for size in dataset_sizes:
            # Use fresh database to avoid interference
            fresh_db = CharactersRAGDB(":memory:", f"scaling_test_{size}")
            
            try:
                # Create dataset
                character_ids = create_large_character_dataset(fresh_db, size)
                
                # Benchmark search performance
                stats = benchmark_function(
                    fetch_relevant_ids_by_keywords,
                    None,
                    fresh_db,
                    DatabaseType.CHARACTER_CARDS,
                    ["fantasy"],
                    num_runs=5
                )
                
                scaling_results.append({
                    'size': size,
                    'mean_time': stats['mean_time'],
                    'results_found': len(stats['result'])
                })
                
            finally:
                fresh_db.close_connection()
        
        # Analyze scaling behavior
        print(f"\nSearch Time Scaling Analysis:")
        for result in scaling_results:
            print(f"Dataset {result['size']:5d}: {result['mean_time']:.4f}s "
                  f"({result['results_found']} results)")
        
        # Calculate scaling factor
        small_time = scaling_results[0]['mean_time']  # 100 characters
        large_time = scaling_results[-1]['mean_time']  # 5000 characters
        size_ratio = scaling_results[-1]['size'] / scaling_results[0]['size']  # 50x
        time_ratio = large_time / small_time
        
        print(f"Dataset size ratio: {size_ratio:.1f}x")
        print(f"Time ratio: {time_ratio:.2f}x")
        
        # Time should scale much better than linearly (ideally logarithmically)
        # With database optimization, 50x data should not take 50x time
        assert time_ratio < size_ratio * 0.5, f"Poor scaling: {time_ratio:.2f}x time for {size_ratio:.1f}x data"
    
    @pytest.mark.benchmark
    def test_tag_variety_impact(self, mem_db):
        """Test performance impact of tag variety and distribution."""
        # Create characters with different tag distributions
        scenarios = [
            ("few_unique_tags", 500, 5),    # 500 chars, 5 unique tags (high overlap)
            ("many_unique_tags", 500, 100), # 500 chars, 100 unique tags (low overlap)
            ("medium_unique_tags", 500, 25) # 500 chars, 25 unique tags (medium overlap)
        ]
        
        performance_results = []
        
        for scenario_name, num_chars, num_unique_tags in scenarios:
            fresh_db = CharactersRAGDB(":memory:", f"tag_variety_{scenario_name}")
            
            try:
                # Create characters with controlled tag variety
                tag_pool = [f"tag_{i}" for i in range(num_unique_tags)]
                
                for i in range(num_chars):
                    # Each character gets 3-5 tags from the pool
                    num_char_tags = 3 + (i % 3)
                    char_tags = [tag_pool[j % num_unique_tags] for j in range(i, i + num_char_tags)]
                    
                    char_data = {
                        "name": f"Character {i}",
                        "description": f"Description {i}",
                        "personality": "Test personality",
                        "scenario": "Test scenario",
                        "tags": json.dumps(char_tags),
                        "first_message": f"Hello from character {i}"
                    }
                    fresh_db.add_character_card(char_data)
                
                # Benchmark search performance
                stats = benchmark_function(
                    fetch_relevant_ids_by_keywords,
                    None,
                    fresh_db,
                    DatabaseType.CHARACTER_CARDS,
                    ["tag_0"],  # Search for first tag (should exist in many characters)
                    num_runs=5
                )
                
                performance_results.append({
                    'scenario': scenario_name,
                    'unique_tags': num_unique_tags,
                    'mean_time': stats['mean_time'],
                    'results_found': len(stats['result'])
                })
                
            finally:
                fresh_db.close_connection()
        
        # Analyze tag variety impact
        print(f"\nTag Variety Impact Analysis:")
        for result in performance_results:
            print(f"{result['scenario']:18}: {result['mean_time']:.4f}s "
                  f"({result['unique_tags']} unique tags, {result['results_found']} results)")
        
        # Performance should be consistent across different tag distributions
        times = [r['mean_time'] for r in performance_results]
        max_time = max(times)
        min_time = min(times)
        time_variance = max_time / min_time
        
        assert time_variance < 3.0, f"High performance variance across tag distributions: {time_variance:.2f}x"


class TestRegressionBenchmarks:
    """Benchmarks to detect performance regressions."""
    
    @pytest.mark.benchmark
    def test_baseline_performance_metrics(self, mem_db):
        """Establish baseline performance metrics for regression detection."""
        # Standard test dataset
        num_characters = 1000
        character_ids = create_large_character_dataset(mem_db, num_characters)
        
        # Standard test scenarios
        test_scenarios = [
            (["fantasy"], "Single common tag"),
            (["featured"], "Single rare tag"), 
            (["fantasy", "sci-fi"], "Two tags"),
            (["nonexistent"], "No results")
        ]
        
        baseline_metrics = {}
        
        for tags, description in test_scenarios:
            stats = benchmark_function(
                fetch_relevant_ids_by_keywords,
                None,
                mem_db,
                DatabaseType.CHARACTER_CARDS,
                tags,
                num_runs=10
            )
            
            baseline_metrics[description] = {
                'mean_time': stats['mean_time'],
                'std_dev': stats['std_dev'],
                'results_count': len(stats['result'])
            }
        
        # Print baseline metrics for future comparison
        print(f"\nBaseline Performance Metrics:")
        for scenario, metrics in baseline_metrics.items():
            print(f"{scenario:20}: {metrics['mean_time']:.4f}s ± {metrics['std_dev']:.4f}s "
                  f"({metrics['results_count']} results)")
        
        # Assert baseline performance standards
        assert baseline_metrics["Single common tag"]['mean_time'] < 0.05, "Baseline performance regression detected"
        assert baseline_metrics["Single rare tag"]['mean_time'] < 0.05, "Baseline performance regression detected"
        assert baseline_metrics["Two tags"]['mean_time'] < 0.1, "Baseline performance regression detected"
        assert baseline_metrics["No results"]['mean_time'] < 0.02, "Baseline performance regression detected"
    
    @pytest.mark.benchmark
    def test_performance_consistency(self, mem_db):
        """Test that performance is consistent across multiple runs."""
        # Create dataset
        character_ids = create_large_character_dataset(mem_db, 500)
        
        # Run the same search many times
        times = []
        for _ in range(50):
            start_time = time.perf_counter()
            result_ids = fetch_relevant_ids_by_keywords(
                None, mem_db, DatabaseType.CHARACTER_CARDS, ["fantasy"]
            )
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Analyze consistency
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate coefficient of variation (relative standard deviation)
        cv = std_dev / mean_time if mean_time > 0 else float('inf')
        
        print(f"\nPerformance Consistency Analysis (50 runs):")
        print(f"Mean time: {mean_time:.4f}s")
        print(f"Std dev: {std_dev:.4f}s")
        print(f"Min time: {min_time:.4f}s")
        print(f"Max time: {max_time:.4f}s")
        print(f"Coefficient of variation: {cv:.3f}")
        
        # Performance should be consistent (CV < 0.3)
        assert cv < 0.3, f"Performance too inconsistent: CV = {cv:.3f}"
        
        # No outliers should be more than 5x the median
        median_time = statistics.median(times)
        assert max_time < median_time * 5, f"Performance outlier detected: {max_time:.4f}s vs median {median_time:.4f}s"