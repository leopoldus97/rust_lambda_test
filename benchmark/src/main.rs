use anyhow::Result;
use aws_config::BehaviorVersion;
use aws_sdk_cloudwatch::Client as CloudWatchClient;
use aws_sdk_lambda::Client as LambdaClient;
use chrono::{DateTime, Utc};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::{Backend, CrosstermBackend},
    crossterm::event as ratatui_event,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Wrap},
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::VecDeque,
    io,
    time::{Duration, Instant, SystemTime},
};
use tui_input::{Input, backend::crossterm::EventHandler};

// Constants for Lambda function names
const RUST_LAMBDA_NAME: &str = "rust_lambda_test-dev-rust";
const NODE_LAMBDA_NAME: &str = "rust_lambda_test-dev-node";

// Data structures for API responses
#[derive(Debug, Deserialize)]
struct MemoryUsage {
    rss: u64,
    #[serde(rename = "heapUsed")]
    heap_used: Option<u64>,
    #[serde(rename = "heapTotal")]
    heap_total: Option<u64>,
    vms: Option<u64>,
    shared: Option<u64>,
    data: Option<u64>,
}

#[derive(Debug)]
struct BenchmarkResult {
    algorithm: String,
    runtime: String,
    execution_time_ms: u64,
    memory_usage: MemoryUsage,
}

// Helper function to parse the response and handle malformed JSON
fn parse_benchmark_result(response_text: &str) -> anyhow::Result<BenchmarkResult> {
    // First, try to parse as JSON value
    let value: Value = serde_json::from_str(response_text)?;

    // Extract fields manually to handle duplicates
    let algorithm = value["algorithm"].as_str().unwrap_or("Unknown").to_string();
    let runtime = value["runtime"].as_str().unwrap_or("Unknown").to_string();
    let execution_time_ms = value["executionTimeMs"].as_u64().unwrap_or(0);

    // Parse memory usage
    let memory_obj = &value["memoryUsage"];
    let memory_usage = MemoryUsage {
        rss: memory_obj["rss"].as_u64().unwrap_or(0),
        heap_used: memory_obj["heapUsed"].as_u64(),
        heap_total: memory_obj["heapTotal"].as_u64(),
        vms: memory_obj["vms"].as_u64(),
        shared: memory_obj["shared"].as_u64(),
        data: memory_obj["data"].as_u64(),
    };

    Ok(BenchmarkResult {
        algorithm,
        runtime,
        execution_time_ms,
        memory_usage,
    })
}

#[derive(Debug, Clone)]
struct BenchmarkData {
    timestamp: DateTime<Utc>,
    rust_time: u64,
    node_time: u64,
    rust_memory: u64,
    node_memory: u64,
    task: String,
    parameters: String,
    // AWS metrics (duration and cold start only)
    rust_cold_start: bool,
    node_cold_start: bool,
    rust_duration: Option<u64>,
    node_duration: Option<u64>,
}

#[derive(Debug, Clone)]
enum AppMode {
    Config,
    Running,
    Results,
}

#[derive(Debug, Clone)]
enum ConfigField {
    Limit,
    Fib,
    Matrix,
    Combined,
}

struct App {
    mode: AppMode,
    config_field: ConfigField,
    limit_input: Input,
    fib_input: Input,
    matrix_input: Input,
    combined_input: Input,
    benchmark_data: VecDeque<BenchmarkData>,
    status_message: String,
    client: Client,
    base_url: String,
    is_running: bool,
    // Auto-test fields
    auto_test_enabled: bool,
    auto_test_interval: Duration,
    last_auto_test: Option<Instant>,
    // AWS clients and toggle
    aws_enabled: bool,
    lambda_client: Option<LambdaClient>,
    cloudwatch_client: Option<CloudWatchClient>,
}

impl App {
    fn new() -> Self {
        let mut app = Self {
            mode: AppMode::Config,
            config_field: ConfigField::Limit,
            limit_input: Input::default(),
            fib_input: Input::default(),
            matrix_input: Input::default(),
            combined_input: Input::default(),
            benchmark_data: VecDeque::new(),
            status_message: "Configure benchmark parameters".to_string(),
            client: Client::new(),
            base_url: "placeholder".to_string(), // TODO: Make this configurable
            is_running: false,
            auto_test_enabled: false,
            auto_test_interval: Duration::from_secs(5), // Default 5 seconds
            last_auto_test: None,
            aws_enabled: false, // Disabled by default
            lambda_client: None,
            cloudwatch_client: None,
        };

        // Set default values (empty initially)
        app.limit_input = Input::new("100000".to_string());
        app.fib_input = Input::new("1000000".to_string());
        app.matrix_input = Input::new("100".to_string());
        app.combined_input = Input::new("50000".to_string());

        app
    }

    async fn init_aws_clients(&mut self) -> Result<()> {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        self.lambda_client = Some(LambdaClient::new(&config));
        self.cloudwatch_client = Some(CloudWatchClient::new(&config));
        Ok(())
    }

    async fn fetch_lambda_metrics(
        &self,
        function_name: &str,
        start_time: DateTime<Utc>,
    ) -> Result<(Option<u64>, bool)> {
        let cloudwatch = match &self.cloudwatch_client {
            Some(client) => client,
            None => return Ok((None, false)),
        };

        let end_time = Utc::now();
        let start_system_time = SystemTime::from(start_time);
        let end_system_time = SystemTime::from(end_time);

        // Build dimension
        let dimension = aws_sdk_cloudwatch::types::Dimension::builder()
            .name("FunctionName")
            .value(function_name)
            .build();

        // Fetch duration only
        let duration_request = cloudwatch
            .get_metric_statistics()
            .namespace("AWS/Lambda")
            .metric_name("Duration")
            .dimensions(dimension)
            .start_time(aws_sdk_cloudwatch::primitives::DateTime::from(
                start_system_time,
            ))
            .end_time(aws_sdk_cloudwatch::primitives::DateTime::from(
                end_system_time,
            ))
            .period(300)
            .statistics(aws_sdk_cloudwatch::types::Statistic::Average)
            .send()
            .await;

        // Check for cold start (simplified logic)
        let _cold_start = start_time.timestamp() > (Utc::now().timestamp() - 300); // Within last 5 minutes

        let duration = if let Ok(response) = duration_request {
            response
                .datapoints()
                .first()
                .and_then(|dp| dp.average())
                .map(|d| d as u64)
        } else {
            // eprintln!("Failed to fetch Lambda metrics: {duration_request:?}");
            None
        };

        Ok((duration, false))
    }

    fn next_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Limit => ConfigField::Fib,
            ConfigField::Fib => ConfigField::Matrix,
            ConfigField::Matrix => ConfigField::Combined,
            ConfigField::Combined => ConfigField::Limit,
        };
    }

    fn prev_field(&mut self) {
        self.config_field = match self.config_field {
            ConfigField::Limit => ConfigField::Combined,
            ConfigField::Fib => ConfigField::Limit,
            ConfigField::Matrix => ConfigField::Fib,
            ConfigField::Combined => ConfigField::Matrix,
        };
    }

    fn get_current_input(&mut self) -> &mut Input {
        match self.config_field {
            ConfigField::Limit => &mut self.limit_input,
            ConfigField::Fib => &mut self.fib_input,
            ConfigField::Matrix => &mut self.matrix_input,
            ConfigField::Combined => &mut self.combined_input,
        }
    }

    fn build_query_params(&self) -> String {
        // Determine task based on which fields have values and current field focus
        let (_task, params) = match self.config_field {
            ConfigField::Limit => (
                "primes",
                vec![
                    "task=primes".to_string(),
                    format!("limit={}", self.limit_input.value()),
                ],
            ),
            ConfigField::Fib => (
                "fibonacci",
                vec![
                    "task=fibonacci".to_string(),
                    format!("fib={}", self.fib_input.value()),
                ],
            ),
            ConfigField::Matrix => (
                "matrix",
                vec![
                    "task=matrix".to_string(),
                    format!("matrix={}", self.matrix_input.value()),
                ],
            ),
            ConfigField::Combined => (
                "combined",
                vec![
                    "task=combined".to_string(),
                    format!("limit={}", self.limit_input.value()),
                    format!("fib={}", self.fib_input.value()),
                    format!("matrix={}", self.matrix_input.value()),
                ],
            ),
        };

        params.join("&")
    }

    fn toggle_aws(&mut self) {
        self.aws_enabled = !self.aws_enabled;
        if self.aws_enabled {
            self.status_message = "AWS duration metrics enabled. Press 'W' to disable".to_string();
        } else {
            self.status_message = "AWS duration metrics disabled. Press 'W' to enable".to_string();
            // Clear AWS clients when disabled
            self.lambda_client = None;
            self.cloudwatch_client = None;
        }
    }

    fn toggle_auto_test(&mut self) {
        self.auto_test_enabled = !self.auto_test_enabled;
        if self.auto_test_enabled {
            self.last_auto_test = Some(Instant::now());
            self.status_message = format!(
                "Auto-test enabled (every {}s). Press 'A' to disable",
                self.auto_test_interval.as_secs()
            );
        } else {
            self.last_auto_test = None;
            self.status_message = "Auto-test disabled".to_string();
        }
    }

    fn should_auto_test(&self) -> bool {
        if !self.auto_test_enabled || self.is_running {
            return false;
        }

        if let Some(last_test) = self.last_auto_test {
            last_test.elapsed() >= self.auto_test_interval
        } else {
            true
        }
    }

    fn update_auto_test_timer(&mut self) {
        if self.auto_test_enabled {
            self.last_auto_test = Some(Instant::now());
        }
    }

    async fn run_benchmark(&mut self) -> Result<()> {
        self.is_running = true;
        self.status_message = "Running benchmark...".to_string();

        // Initialize AWS clients if not already done and AWS is enabled
        if self.aws_enabled && (self.lambda_client.is_none() || self.cloudwatch_client.is_none()) {
            let _ = self.init_aws_clients().await;
        }

        let start_time = Utc::now();
        let query_params = self.build_query_params();
        let rust_url = format!("{}/rust?{}", self.base_url, query_params);
        let node_url = format!("{}/node?{}", self.base_url, query_params);

        // Run benchmarks
        let rust_start = Instant::now();
        let rust_response = self.client.get(&rust_url).send().await?;
        let _rust_duration = rust_start.elapsed();

        let node_start = Instant::now();
        let node_response = self.client.get(&node_url).send().await?;
        let _node_duration = node_start.elapsed();

        if rust_response.status().is_success() && node_response.status().is_success() {
            let rust_text = rust_response.text().await?;
            let node_text = node_response.text().await?;

            let rust_result = parse_benchmark_result(&rust_text)?;
            let node_result = parse_benchmark_result(&node_text)?;

            // Fetch AWS metrics for both functions
            let (rust_aws_duration, rust_cold_start) = if self.aws_enabled {
                self.fetch_lambda_metrics(RUST_LAMBDA_NAME, start_time)
                    .await
                    .unwrap_or((None, false))
            } else {
                (None, false)
            };

            // eprintln!(
            //     "Rust Lambda Metrics: {:?}",
            //     (rust_aws_duration, rust_cold_start)
            // );

            let (node_aws_duration, node_cold_start) = if self.aws_enabled {
                self.fetch_lambda_metrics(NODE_LAMBDA_NAME, start_time)
                    .await
                    .unwrap_or((None, false))
            } else {
                (None, false)
            };

            let benchmark_data = BenchmarkData {
                timestamp: start_time,
                rust_time: rust_result.execution_time_ms,
                node_time: node_result.execution_time_ms,
                rust_memory: rust_result.memory_usage.rss,
                node_memory: rust_result.memory_usage.rss
                    + node_result.memory_usage.heap_used.unwrap_or(0),
                task: match self.config_field {
                    ConfigField::Limit => "primes".to_string(),
                    ConfigField::Fib => "fibonacci".to_string(),
                    ConfigField::Matrix => "matrix".to_string(),
                    ConfigField::Combined => "combined".to_string(),
                },
                parameters: query_params,
                rust_cold_start,
                node_cold_start,
                rust_duration: rust_aws_duration,
                node_duration: node_aws_duration,
            };

            self.benchmark_data.push_back(benchmark_data);
            if self.benchmark_data.len() > 50 {
                self.benchmark_data.pop_front();
            }

            self.status_message = format!(
                "Benchmark completed! Rust: {}ms, Node: {}ms",
                rust_result.execution_time_ms, node_result.execution_time_ms
            );
        } else {
            self.status_message = format!(
                "Benchmark failed: API error. Rust: {rust_url} -> {}, Node: {node_url} -> {}",
                rust_response.status(),
                node_response.status()
            );
        }

        self.is_running = false;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run
    let mut app = App::new();
    let res: std::result::Result<(), anyhow::Error> = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

async fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        // Check for auto-test
        if app.should_auto_test() && matches!(app.mode, AppMode::Results) {
            app.mode = AppMode::Running;
            let _ = app.run_benchmark().await;
            app.update_auto_test_timer();
            app.mode = AppMode::Results;
        }

        if event::poll(Duration::from_millis(100))? {
            if let ratatui_event::Event::Key(key) = ratatui_event::read()? {
                if key.kind == ratatui_event::KeyEventKind::Press {
                    match app.mode {
                        AppMode::Config => match key.code {
                            ratatui_event::KeyCode::Char('q') => return Ok(()),
                            ratatui_event::KeyCode::Tab => app.next_field(),
                            ratatui_event::KeyCode::BackTab => app.prev_field(),
                            ratatui_event::KeyCode::Char('w')
                            | ratatui_event::KeyCode::Char('W') => {
                                app.toggle_aws();
                            }
                            ratatui_event::KeyCode::Enter => {
                                app.mode = AppMode::Running;
                                let _ = app.run_benchmark().await;
                                app.mode = AppMode::Results;
                            }
                            _ => {
                                // Don't handle input for Combined field since it's read-only
                                if !matches!(app.config_field, ConfigField::Combined) {
                                    app.get_current_input()
                                        .handle_event(&ratatui_event::Event::Key(key));
                                }
                            }
                        },
                        AppMode::Running => {
                            // Can't do anything while running
                        }
                        AppMode::Results => match key.code {
                            ratatui_event::KeyCode::Char('q') => return Ok(()),
                            ratatui_event::KeyCode::Char('r') => {
                                app.mode = AppMode::Running;
                                let _ = app.run_benchmark().await;
                                app.update_auto_test_timer();
                                app.mode = AppMode::Results;
                            }
                            ratatui_event::KeyCode::Char('c') => {
                                app.mode = AppMode::Config;
                            }
                            ratatui_event::KeyCode::Char('x') => {
                                app.benchmark_data.clear();
                                app.status_message = "Cleared benchmark history".to_string();
                            }
                            ratatui_event::KeyCode::Char('a')
                            | ratatui_event::KeyCode::Char('A') => {
                                app.toggle_auto_test();
                            }
                            ratatui_event::KeyCode::Char('w')
                            | ratatui_event::KeyCode::Char('W') => {
                                app.toggle_aws();
                            }
                            _ => {}
                        },
                    }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    // Check minimum terminal size
    const MIN_WIDTH: u16 = 120;
    const MIN_HEIGHT: u16 = 30;

    let size = f.area();
    if size.width < MIN_WIDTH || size.height < MIN_HEIGHT {
        let warning = Paragraph::new(format!(
            "Terminal too small!\nMinimum size required: {}x{}\nCurrent size: {}x{}\nPlease resize your terminal.",
            MIN_WIDTH, MIN_HEIGHT, size.width, size.height
        ))
        .block(Block::default().borders(Borders::ALL).title("Warning"))
        .style(Style::default().fg(Color::Red))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });

        f.render_widget(warning, size);
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Length(12), // Config (increased to accommodate combined field)
            Constraint::Min(10),    // Charts/Results
            Constraint::Length(3),  // Status
        ])
        .split(f.area());

    // Title
    let title = Paragraph::new("🚀 Lambda Benchmark Tool 🚀")
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Config section
    render_config_section(f, app, chunks[1]);

    // Results section
    match app.mode {
        AppMode::Config | AppMode::Running => {
            render_instructions(f, chunks[2]);
        }
        AppMode::Results => {
            render_results_section(f, app, chunks[2]);
        }
    }

    // Status
    let status_color = match app.mode {
        AppMode::Running => Color::Yellow,
        _ => Color::Green,
    };

    let status = Paragraph::new(app.status_message.clone())
        .style(Style::default().fg(status_color))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL).title("Status"));
    f.render_widget(status, chunks[3]);
}

fn render_config_section(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Configuration");

    let inner = block.inner(area);
    f.render_widget(block, area);

    let config_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Current task display
            Constraint::Length(2), // Limit
            Constraint::Length(2), // Fib
            Constraint::Length(2), // Matrix
            Constraint::Length(2), // Combined
            Constraint::Length(4), // Instructions
        ])
        .split(inner);

    // Display current task (read-only)
    let current_task = match app.config_field {
        ConfigField::Limit => "Current Task: Primes",
        ConfigField::Fib => "Current Task: Fibonacci",
        ConfigField::Matrix => "Current Task: Matrix Multiplication",
        ConfigField::Combined => "Current Task: Combined (All Tasks)",
    };

    let task_paragraph = Paragraph::new(current_task)
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::NONE));
    f.render_widget(task_paragraph, config_chunks[0]);

    let fields = [
        ("Limit (for primes)", &app.limit_input, ConfigField::Limit),
        ("Fibonacci N", &app.fib_input, ConfigField::Fib),
        ("Matrix Size", &app.matrix_input, ConfigField::Matrix),
        (
            "Combined (uses all above)",
            &app.combined_input,
            ConfigField::Combined,
        ),
    ];

    for (i, (label, input, field)) in fields.iter().enumerate() {
        let style = if std::mem::discriminant(&app.config_field) == std::mem::discriminant(field)
            && matches!(app.mode, AppMode::Config)
        {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        // Special handling for Combined field to show all parameters
        let display_text = if matches!(field, ConfigField::Combined) {
            format!(
                "{}: limit={}, fib={}, matrix={}",
                label,
                app.limit_input.value(),
                app.fib_input.value(),
                app.matrix_input.value()
            )
        } else {
            format!("{}: {}", label, input.value())
        };

        let paragraph = Paragraph::new(display_text)
            .style(style)
            .block(Block::default().borders(Borders::NONE));
        f.render_widget(paragraph, config_chunks[i + 1]); // +1 to account for task display
    }

    // Instructions
    let aws_status = if app.aws_enabled { "ON" } else { "OFF" };
    let instructions = vec![
        Line::from(format!(
            "Tab/Shift+Tab: Navigate | Enter: Run | W: AWS {} | Q: Quit",
            aws_status
        )),
        Line::from(
            "Tasks: Limit→Primes, Fib→Fibonacci, Matrix→Matrix, Combined→All (uses all values)",
        ),
    ];
    let instructions_paragraph = Paragraph::new(instructions)
        .style(Style::default().fg(Color::Gray))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
    f.render_widget(instructions_paragraph, config_chunks[4]);
}

fn render_instructions(f: &mut Frame, area: Rect) {
    let instructions = vec![
        Line::from("Welcome to Lambda Benchmark Tool!"),
        Line::from(""),
        Line::from("1. Navigate between parameter fields using Tab/Shift+Tab"),
        Line::from("2. The task will automatically change based on the selected field"),
        Line::from("3. Enter values for the parameters you want to test"),
        Line::from("4. Press Enter to run the benchmark"),
        Line::from("5. View results and charts after running"),
        Line::from(""),
        Line::from("Tasks automatically selected:"),
        Line::from("  • Limit field → Primes task"),
        Line::from("  • Fibonacci N field → Fibonacci task"),
        Line::from("  • Matrix Size field → Matrix Multiplication task"),
        Line::from("  • Combined Limit field → Combined task (runs all tasks with set values)"),
        Line::from(""),
        Line::from("Press 'W' to toggle AWS metrics (cold starts, duration)"),
    ];

    let paragraph = Paragraph::new(instructions)
        .block(Block::default().borders(Borders::ALL).title("Instructions"))
        .wrap(Wrap { trim: true });
    f.render_widget(paragraph, area);
}

fn render_results_section(f: &mut Frame, app: &App, area: Rect) {
    if app.benchmark_data.is_empty() {
        let msg = Paragraph::new("No benchmark data yet. Run a benchmark to see results!")
            .block(Block::default().borders(Borders::ALL).title("Results"))
            .alignment(Alignment::Center);
        f.render_widget(msg, area);
        return;
    }

    let results_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    // Performance chart
    render_performance_chart(f, app, results_chunks[0]);

    // Stats and controls
    render_stats_panel(f, app, results_chunks[1]);
}

fn render_performance_chart(f: &mut Frame, app: &App, area: Rect) {
    let chart_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Execution time chart
    let time_data: Vec<(f64, f64)> = app
        .benchmark_data
        .iter()
        .enumerate()
        .map(|(i, data)| (i as f64, data.rust_time as f64))
        .collect();

    let node_time_data: Vec<(f64, f64)> = app
        .benchmark_data
        .iter()
        .enumerate()
        .map(|(i, data)| (i as f64, data.node_time as f64))
        .collect();

    let max_time = app
        .benchmark_data
        .iter()
        .map(|d| d.rust_time.max(d.node_time))
        .max()
        .unwrap_or(1) as f64;

    let time_datasets = vec![
        Dataset::default()
            .name("Rust")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Red))
            .graph_type(GraphType::Line)
            .data(&time_data),
        Dataset::default()
            .name("Node.js")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Green))
            .graph_type(GraphType::Line)
            .data(&node_time_data),
    ];

    let time_chart = Chart::new(time_datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Execution Time (ms)"),
        )
        .x_axis(
            Axis::default()
                .title("Benchmark Run")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, app.benchmark_data.len() as f64]),
        )
        .y_axis(
            Axis::default()
                .title("Time (ms)")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_time * 1.1]),
        );
    f.render_widget(time_chart, chart_chunks[0]);

    // Memory usage chart
    let rust_memory_data: Vec<(f64, f64)> = app
        .benchmark_data
        .iter()
        .enumerate()
        .map(|(i, data)| (i as f64, data.rust_memory as f64 / 1024.0 / 1024.0)) // Convert to MB
        .collect();

    let node_memory_data: Vec<(f64, f64)> = app
        .benchmark_data
        .iter()
        .enumerate()
        .map(|(i, data)| (i as f64, data.node_memory as f64 / 1024.0 / 1024.0)) // Convert to MB
        .collect();

    let max_memory = app
        .benchmark_data
        .iter()
        .map(|d| d.rust_memory.max(d.node_memory) as f64 / 1024.0 / 1024.0)
        .fold(0.0, f64::max);

    let memory_datasets = vec![
        Dataset::default()
            .name("Rust")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Red))
            .graph_type(GraphType::Line)
            .data(&rust_memory_data),
        Dataset::default()
            .name("Node.js")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Green))
            .graph_type(GraphType::Line)
            .data(&node_memory_data),
    ];

    let memory_chart = Chart::new(memory_datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Memory Usage (MB)"),
        )
        .x_axis(
            Axis::default()
                .title("Benchmark Run")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, app.benchmark_data.len() as f64]),
        )
        .y_axis(
            Axis::default()
                .title("Memory (MB)")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_memory * 1.1]),
        );
    f.render_widget(memory_chart, chart_chunks[1]);
}

fn render_stats_panel(f: &mut Frame, app: &App, area: Rect) {
    let stats_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),    // Latest results
            Constraint::Min(6),     // Average stats
            Constraint::Length(10), // Controls (increased for AWS toggle)
        ])
        .split(area);

    // Latest results
    if let Some(latest) = app.benchmark_data.back() {
        let mut latest_lines = vec![
            Line::from(format!("Task: {}", latest.task)),
            Line::from(format!("Parameters: {}", latest.parameters)),
            Line::from(""),
            Line::from(vec![
                Span::styled("Rust: ", Style::default().fg(Color::Red)),
                Span::raw(format!("{}ms", latest.rust_time)),
                if latest.rust_cold_start {
                    Span::styled(" (COLD)", Style::default().fg(Color::Blue))
                } else {
                    Span::raw("")
                },
            ]),
            Line::from(vec![
                Span::styled("Node: ", Style::default().fg(Color::Green)),
                Span::raw(format!("{}ms", latest.node_time)),
                if latest.node_cold_start {
                    Span::styled(" (COLD)", Style::default().fg(Color::Blue))
                } else {
                    Span::raw("")
                },
            ]),
            Line::from(vec![
                Span::styled("RAM Rust: ", Style::default().fg(Color::Red)),
                Span::raw(format!(
                    "{:.1} MB",
                    latest.rust_memory as f64 / 1024.0 / 1024.0
                )),
            ]),
            Line::from(vec![
                Span::styled("RAM Node: ", Style::default().fg(Color::Green)),
                Span::raw(format!(
                    "{:.1} MB",
                    latest.node_memory as f64 / 1024.0 / 1024.0
                )),
            ]),
        ];

        // Add AWS duration metrics if available
        if let Some(rust_duration) = latest.rust_duration {
            latest_lines.push(Line::from(vec![
                Span::styled("AWS Rust Duration: ", Style::default().fg(Color::Magenta)),
                Span::raw(format!("{:.1} ms", rust_duration as f64)),
            ]));
        }

        if let Some(node_duration) = latest.node_duration {
            latest_lines.push(Line::from(vec![
                Span::styled("AWS Node Duration: ", Style::default().fg(Color::Magenta)),
                Span::raw(format!("{:.1} ms", node_duration as f64)),
            ]));
        }

        latest_lines.push(Line::from(format!(
            "Winner: {}",
            if latest.rust_time < latest.node_time {
                "🦀 Rust"
            } else {
                "🟢 Node.js"
            }
        )));

        let latest_paragraph = Paragraph::new(latest_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Latest Result"),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(latest_paragraph, stats_chunks[0]);
    }

    // Average statistics
    if !app.benchmark_data.is_empty() {
        let avg_rust_time: f64 = app
            .benchmark_data
            .iter()
            .map(|d| d.rust_time as f64)
            .sum::<f64>()
            / app.benchmark_data.len() as f64;
        let avg_node_time: f64 = app
            .benchmark_data
            .iter()
            .map(|d| d.node_time as f64)
            .sum::<f64>()
            / app.benchmark_data.len() as f64;

        let avg_rust_memory: f64 = app
            .benchmark_data
            .iter()
            .map(|d| d.rust_memory as f64 / 1024.0 / 1024.0)
            .sum::<f64>()
            / app.benchmark_data.len() as f64;
        let avg_node_memory: f64 = app
            .benchmark_data
            .iter()
            .map(|d| d.node_memory as f64 / 1024.0 / 1024.0)
            .sum::<f64>()
            / app.benchmark_data.len() as f64;

        let avg_text = vec![
            Line::from(format!("Runs: {}", app.benchmark_data.len())),
            Line::from(format!("Avg Rust: {:.1}ms", avg_rust_time)),
            Line::from(format!("Avg Node: {:.1}ms", avg_node_time)),
            Line::from(format!("Avg RAM Rust: {:.1}MB", avg_rust_memory)),
            Line::from(format!("Avg RAM Node: {:.1}MB", avg_node_memory)),
            Line::from(format!(
                "Rust wins: {}%",
                (app.benchmark_data
                    .iter()
                    .filter(|d| d.rust_time < d.node_time)
                    .count() as f64
                    / app.benchmark_data.len() as f64
                    * 100.0) as u32
            )),
        ];

        let avg_paragraph = Paragraph::new(avg_text)
            .block(Block::default().borders(Borders::ALL).title("Statistics"));
        f.render_widget(avg_paragraph, stats_chunks[1]);
    }

    // Controls
    let auto_test_status = if app.auto_test_enabled {
        format!("Auto-test: ON ({}s)", app.auto_test_interval.as_secs())
    } else {
        "Auto-test: OFF".to_string()
    };

    let aws_status = if app.aws_enabled {
        "AWS: ON".to_string()
    } else {
        "AWS: OFF".to_string()
    };

    let controls_text = vec![
        Line::from("R: Run Again"),
        Line::from("A: Toggle Auto-test"),
        Line::from(auto_test_status),
        Line::from("W: Toggle AWS"),
        Line::from(aws_status),
        Line::from("C: Configure"),
        Line::from("X: Clear History"),
        Line::from("Q: Quit"),
    ];

    let controls_paragraph = Paragraph::new(controls_text)
        .block(Block::default().borders(Borders::ALL).title("Controls"))
        .style(Style::default().fg(Color::Gray));
    f.render_widget(controls_paragraph, stats_chunks[2]);
}
