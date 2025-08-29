use lambda_runtime::{Error, LambdaEvent, run, service_fn};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Deserialize)]
struct Request {
    #[serde(rename = "queryStringParameters")]
    query_string_parameters: Option<HashMap<String, String>>,
}

#[derive(Serialize)]
struct MemoryUsage {
    rss: u64,    // Resident Set Size (physical memory currently used)
    vms: u64,    // Virtual Memory Size (total virtual memory used)
    shared: u64, // Shared memory
    data: u64,   // Data/heap segment
}

#[derive(Serialize)]
struct BenchmarkResult {
    algorithm: String,
    runtime: String,
    #[serde(rename = "executionTimeMs")]
    execution_time_ms: u128,
    #[serde(rename = "memoryUsage")]
    memory_usage: MemoryUsage,
    #[serde(flatten)]
    details: Value,
}

// Function to get current memory usage
#[cfg(target_os = "linux")]
fn get_memory_usage() -> MemoryUsage {
    match procfs::process::Process::myself() {
        Ok(me) => {
            if let Ok(stat) = me.stat() {
                // Convert from pages to bytes (page size is typically 4096 bytes)
                let page_size = 4096u64;
                MemoryUsage {
                    rss: stat.rss * page_size,
                    vms: stat.vsize,
                    shared: stat.rsslim,
                    data: stat.vsize.saturating_sub(stat.rss * page_size),
                }
            } else {
                // Fallback if we can't read detailed stats
                MemoryUsage {
                    rss: 0,
                    vms: 0,
                    shared: 0,
                    data: 0,
                }
            }
        }
        Err(_) => {
            // Fallback for when procfs is not available (shouldn't happen on Linux)
            MemoryUsage {
                rss: 0,
                vms: 0,
                shared: 0,
                data: 0,
            }
        }
    }
}

// Fallback function for non-Linux systems (development on macOS/Windows)
#[cfg(not(target_os = "linux"))]
fn get_memory_usage() -> MemoryUsage {
    // Return placeholder values for non-Linux systems
    // In production (Lambda), this will use the Linux version above
    MemoryUsage {
        rss: 0,
        vms: 0,
        shared: 0,
        data: 0,
    }
}

// CPU-intensive prime number calculation using Sieve of Eratosthenes
fn calculate_primes(limit: usize) -> (Vec<usize>, usize) {
    if limit < 2 {
        return (Vec::new(), 0);
    }

    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 {
        is_prime[1] = false;
    }

    let mut p = 2;
    while p * p <= limit {
        if is_prime[p] {
            let mut i = p * p;
            while i <= limit {
                is_prime[i] = false;
                i += p;
            }
        }
        p += 1;
    }

    let primes: Vec<usize> = (2..=limit).filter(|&i| is_prime[i]).collect();

    let count = primes.len();
    (primes, count)
}

// CPU-intensive Fibonacci calculation
fn fibonacci_iterative(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }

    let mut a = 0u64;
    let mut b = 1u64;

    for _ in 2..=n {
        let temp = a.saturating_add(b);
        a = b;
        b = temp;
    }

    b
}

// Matrix multiplication for additional CPU load
fn matrix_multiply(size: usize) -> Vec<Vec<f64>> {
    use rand::Rng;
    let mut rng = rand::rng();

    // Generate random matrices
    let a: Vec<Vec<f64>> = (0..size)
        .map(|_| (0..size).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect();

    let b: Vec<Vec<f64>> = (0..size)
        .map(|_| (0..size).map(|_| rng.random_range(0.0..1.0)).collect())
        .collect();

    // Multiply matrices
    let mut result = vec![vec![0.0; size]; size];

    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Value, Error> {
    let start_time = Instant::now();
    let start_memory = get_memory_usage();

    let query_params = event.payload.query_string_parameters.unwrap_or_default();

    let limit: usize = query_params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100000);

    let fib_number: u64 = query_params
        .get("fib")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000000);

    let matrix_size: usize = query_params
        .get("matrix")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let task = query_params
        .get("task")
        .map(|s| s.as_str())
        .unwrap_or("primes");

    let result = match task {
        "primes" => {
            let (primes, count) = calculate_primes(limit);
            let last_prime = primes.last().copied().unwrap_or(0);

            json!({
                "algorithm": "Sieve of Eratosthenes",
                "requestedLimit": limit,
                "primesFound": count,
                "lastPrime": last_prime,
                "samplePrimes": {
                    "first10": primes.iter().take(10).collect::<Vec<_>>(),
                    "last10": primes.iter().rev().take(10).rev().collect::<Vec<_>>()
                }
            })
        }
        "fibonacci" => {
            let fib_result = fibonacci_iterative(fib_number);
            json!({
                "algorithm": "Iterative Fibonacci",
                "requestedNumber": fib_number,
                "result": fib_result
            })
        }
        "matrix" => {
            let matrix_result = matrix_multiply(matrix_size);
            json!({
                "algorithm": "Matrix Multiplication",
                "matrixSize": matrix_size,
                "resultSample": {
                    "topLeft": matrix_result[0][0],
                    "bottomRight": matrix_result[matrix_size - 1][matrix_size - 1]
                }
            })
        }
        "combined" => {
            let combined_start = Instant::now();

            let safe_limit = std::cmp::min(limit, 50000);
            let safe_fib = std::cmp::min(fib_number, 100000);
            let safe_matrix = std::cmp::min(matrix_size, 50);

            let (_, prime_count) = calculate_primes(safe_limit);
            let fib_result = fibonacci_iterative(safe_fib);
            let _matrix_result = matrix_multiply(safe_matrix);

            json!({
                "algorithm": "Combined CPU Tasks",
                "tasks": {
                    "primes": { "count": prime_count, "limit": safe_limit },
                    "fibonacci": { "number": safe_fib, "result": fib_result },
                    "matrix": { "size": safe_matrix }
                },
                "combinedExecutionTime": combined_start.elapsed().as_millis()
            })
        }
        _ => {
            return Ok(json!({
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json!({
                    "error": "Bad Request",
                    "message": format!("Unknown task: {}", task),
                    "runtime": "Rust"
                }).to_string()
            }));
        }
    };

    let execution_time = start_time.elapsed();
    let end_memory = get_memory_usage();

    // Calculate memory usage delta (end - start)
    let memory_usage = MemoryUsage {
        rss: end_memory.rss.saturating_sub(start_memory.rss),
        vms: end_memory.vms.saturating_sub(start_memory.vms),
        shared: end_memory.shared.saturating_sub(start_memory.shared),
        data: end_memory.data.saturating_sub(start_memory.data),
    };

    let benchmark_result = BenchmarkResult {
        algorithm: result["algorithm"]
            .as_str()
            .unwrap_or("Unknown")
            .to_string(),
        runtime: "Rust".to_string(),
        execution_time_ms: execution_time.as_millis(),
        memory_usage,
        details: result,
    };

    Ok(json!({
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        },
        "body": serde_json::to_string(&benchmark_result)?
    }))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    run(service_fn(function_handler)).await
}
