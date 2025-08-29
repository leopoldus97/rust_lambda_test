import type { APIGatewayProxyEvent, APIGatewayProxyResult } from "aws-lambda";

interface BenchmarkResult {
  algorithm: string;
  runtime: string;
  executionTimeMs: number;
  memoryUsage?: {
    rss: number;
    heapUsed: number;
    heapTotal: number;
  };
  
}

// CPU-intensive prime number calculation using Sieve of Eratosthenes
function calculatePrimes(limit: number): { primes: number[]; count: number } {
  if (limit < 2) return { primes: [], count: 0 };

  // Create boolean array "prime[0..limit]" and initialize all entries as true
  const prime = new Array(limit + 1).fill(true);
  prime[0] = prime[1] = false;

  for (let p = 2; p * p <= limit; p++) {
    // If prime[p] is not changed, then it is a prime
    if (prime[p]) {
      // Update all multiples of p
      for (let i = p * p; i <= limit; i += p) {
        prime[i] = false;
      }
    }
  }

  // Collect all prime numbers
  const primes: number[] = [];
  for (let i = 2; i <= limit; i++) {
    if (prime[i]) {
      primes.push(i);
    }
  }

  return { primes, count: primes.length };
}

// Additional CPU-intensive task: Calculate Fibonacci sequence
function fibonacciIterative(n: number): number {
  if (n <= 1) return n;

  let a = 0;
  let b = 1;

  for (let i = 2; i <= n; i++) {
    const temp = a + b;
    a = b;
    b = temp;
  }

  return b;
}

// Matrix multiplication for additional CPU load
function matrixMultiply(size: number): number[][] {
  const a = Array(size)
    .fill(0)
    .map(() =>
      Array(size)
        .fill(0)
        .map(() => Math.random())
    );
  const b = Array(size)
    .fill(0)
    .map(() =>
      Array(size)
        .fill(0)
        .map(() => Math.random())
    );
  const result = Array(size)
    .fill(0)
    .map(() => Array(size).fill(0));

  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      for (let k = 0; k < size; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

export const handler = async (
  event: APIGatewayProxyEvent
): Promise<APIGatewayProxyResult> => {
  const startTime = Date.now();
  const startMemory = process.memoryUsage();

  try {
    // Parse query parameters
    const queryParams = event.queryStringParameters || {};
    const limit = parseInt(queryParams.limit || "100000", 10);
    const fibNumber = parseInt(queryParams.fib || "1000000", 10);
    const matrixSize = parseInt(queryParams.matrix || "100", 10);
    const task = queryParams.task || "primes";

    let result: any = {};

    switch (task) {
      case "primes":
        const primeResult = calculatePrimes(limit);
        result = {
          algorithm: "Sieve of Eratosthenes",
          requestedLimit: limit,
          primesFound: primeResult.count,
          lastPrime: primeResult.primes[primeResult.primes.length - 1] || 0,
          // Only include first and last few primes to avoid large response
          samplePrimes: {
            first10: primeResult.primes.slice(0, 10),
            last10: primeResult.primes.slice(-10),
          },
        };
        break;

      case "fibonacci":
        const fibResult = fibonacciIterative(fibNumber);
        result = {
          algorithm: "Iterative Fibonacci",
          requestedNumber: fibNumber,
          result:
            fibResult > Number.MAX_SAFE_INTEGER
              ? "Number too large for JS"
              : fibResult,
        };
        break;

      case "matrix":
        const matrixResult = matrixMultiply(matrixSize);
        result = {
          algorithm: "Matrix Multiplication",
          matrixSize: matrixSize,
          resultSample: {
            topLeft: matrixResult[0][0],
            bottomRight: matrixResult[matrixSize - 1][matrixSize - 1],
          },
        };
        break;

      case "combined":
        // Run all tasks for comprehensive benchmarking
        const combinedStart = Date.now();

        const primes = calculatePrimes(Math.min(limit, 50000));
        const fib = fibonacciIterative(Math.min(fibNumber, 100000));
        const matrix = matrixMultiply(Math.min(matrixSize, 50));

        result = {
          algorithm: "Combined CPU Tasks",
          tasks: {
            primes: { count: primes.count, limit },
            fibonacci: { number: fibNumber, result: fib },
            matrix: { size: matrixSize },
          },
          combinedExecutionTime: Date.now() - combinedStart,
        };
        break;

      default:
        throw new Error(`Unknown task: ${task}`);
    }

    const endTime = Date.now();
    const endMemory = process.memoryUsage();

    const benchmarkResult: BenchmarkResult = {
      algorithm: result.algorithm,
      runtime: "Node.js",
      executionTimeMs: endTime - startTime,
      memoryUsage: {
        rss: endMemory.rss - startMemory.rss,
        heapUsed: endMemory.heapUsed - startMemory.heapUsed,
        heapTotal: endMemory.heapTotal - startMemory.heapTotal,
      },
      ...result,
    };

    return {
      statusCode: 200,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      },
      body: JSON.stringify(benchmarkResult, null, 2),
    };
  } catch (error) {
    const endTime = Date.now();

    return {
      statusCode: 500,
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
      body: JSON.stringify({
        error: "Internal Server Error",
        message: error instanceof Error ? error.message : "Unknown error",
        runtime: "Node.js",
        executionTimeMs: endTime - startTime,
      }),
    };
  }
};
