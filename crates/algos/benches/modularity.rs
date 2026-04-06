use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, SamplingMode};
use graph::modularity::{local_modularity_optimization, modularity};
use graph_builder::{CsrLayout, GraphBuilder, UndirectedCsrGraph};
use polars::prelude::{DataFrame, LazyFrame, PlRefPath};
use std::iter::zip;
use std::time::Duration;

fn bench_example(c: &mut Criterion) {
    let mut group = c.benchmark_group("example");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(1))
        .warm_up_time(Duration::from_millis(100))
        .sampling_mode(SamplingMode::Flat);

    group.bench_function("run", |b| {
        b.iter_batched(
            || {
                // let mut rng = rand::thread_rng();
                // let node_count = 10_000;
                // let edge_count = 10 * node_count;
                // let graph: UndirectedCsrGraph<usize, _, f64> = GraphBuilder::new()
                //     .csr_layout(CsrLayout::Deduplicated)
                //     .edges_with_values((0..edge_count).into_iter().map(|_| {
                //         (
                //             rng.gen_range(0..node_count),
                //             rng.gen_range(0..node_count),
                //             1.,
                //         )
                //     }))
                //     .build();
                let df: DataFrame = LazyFrame::scan_parquet(
                    PlRefPath::from(
                        "/Users/alfred/proj/louvain/graphs/power_law_n10000_d10.parquet",
                    ),
                    Default::default(),
                )
                .unwrap()
                .collect()
                .unwrap();

                let edges_with_values: Vec<(u64, u64, f64)> = zip(
                    df.column("sourceNodeId")
                        .unwrap()
                        .i64()
                        .unwrap()
                        .into_no_null_iter(),
                    df.column("targetNodeId")
                        .unwrap()
                        .i64()
                        .unwrap()
                        .into_no_null_iter(),
                )
                .map(|(s, t)| (s as u64, t as u64, 1.))
                .collect();
                println!("read!");
                let graph: UndirectedCsrGraph<u64, _, f64> = GraphBuilder::new()
                    .csr_layout(CsrLayout::Deduplicated)
                    .edges_with_values(edges_with_values)
                    .build();
                println!("built!");
                graph
            },
            |graph| {
                black_box({
                    let communities = local_modularity_optimization(&graph, 10, 0.);
                    let q = modularity(&graph, &communities);
                    println!("Modularity: {}", q)
                });
            },
            BatchSize::PerIteration,
        );
    });
}

criterion_group!(benches, bench_example);
criterion_main!(benches);
