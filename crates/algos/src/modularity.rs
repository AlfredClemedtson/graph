use graph_builder::prelude::*;
use std::collections::{HashMap, HashSet};

const MAX_ITERATIONS: u32 = 20;

pub fn local_modularity_optimization<NI, G>(graph: &G) -> Vec<usize>
where
    NI: Idx,
    G: Graph<NI> + UndirectedNeighborsWithValues<NI, f64>,
{
    let node_count = graph.node_count().index();
    let weighted_degree = (0..node_count) //can be parallelized
        .map(|u| {
            graph
                .neighbors_with_values(NI::new(u))
                .map(|Target { target: _, value }| value)
                .sum()
        })
        .collect::<Vec<_>>();
    let mut weighted_degree_of_community = weighted_degree.clone();
    let total_edge_weight: f64 = weighted_degree.iter().sum();
    let mut communities: Vec<_> = (0..node_count).collect();
    let mut counter = 0;
    let mut acc_diffs = 0.;
    let q = modularity(graph, &communities);
    println!("Iterations: {}, Modularity: {}, Diffs: {}, Acc diffs: {}", counter, q, 0., acc_diffs);
    for _ in 0..MAX_ITERATIONS {
        let mut diffs = 0.;
        for u in 0..node_count {
            let old_c = communities[u];
            match best_community(
                graph,
                &communities,
                &weighted_degree_of_community,
                &weighted_degree,
                total_edge_weight,
                NI::new(u)
            ) {
                Some((new_c, diff)) => {
                    diffs += diff;
                    communities[u] = new_c;
                    weighted_degree_of_community[new_c] += weighted_degree[u];
                    weighted_degree_of_community[old_c] -= weighted_degree[u];
                },
                None => {}
            }
        }
        counter += 1;
        let q = modularity(graph, &communities);
        acc_diffs += diffs;
        println!("Iterations: {}, Modularity: {}, Diffs: {}, Acc diffs: {}", counter, q, diffs, acc_diffs);
    }
    communities
}

fn modularity<NI, G>(graph: &G, communities: &Vec<usize>) -> f64
where
    NI: Idx,
    G: Graph<NI> + UndirectedNeighborsWithValues<NI, f64>,
{
    let grouped_communities = communities.iter().enumerate().fold(
        HashMap::<usize, HashSet<usize>>::new(),
        |mut sets, (u, c)| {
            if !sets.contains_key(c) {
                sets.insert(*c, HashSet::new());
            }
            sets.get_mut(c).unwrap().insert(u);
            sets
        },
    );
    let node_count = graph.node_count().index();
    let total_edge_weight: f64 = (0..node_count)
        .map(|u| {
            graph
                .neighbors_with_values(NI::new(u))
                .map(
                    |&Target {
                         target: _,
                         value: weight,
                     }| weight,
                )
                .sum::<f64>()
        })
        .sum();
    grouped_communities
        .iter()
        .map(|(c, nodes)| {
            let mut sum_in_c = 0.;
            let mut sum_tot = 0.;
            for u in nodes {
                for Target {
                    target: v,
                    value: weight,
                } in graph.neighbors_with_values(NI::new(*u))
                {
                    sum_tot += weight;
                    if communities[v.index()] == *c {
                        sum_in_c += weight;
                    }
                }
            }
            sum_in_c / total_edge_weight - (sum_tot / total_edge_weight).powi(2)
        })
        .sum()
}

fn best_community<NI, G>(
    graph: &G,
    communities: &Vec<usize>,
    sum_k_c: &Vec<f64>,
    k: &Vec<f64>,
    sum_k: f64,
    u: NI,
) -> Option<(usize, f64)>
where
    NI: Idx,
    G: Graph<NI> + UndirectedNeighborsWithValues<NI, f64>,
{
    let old_c = communities[u.index()];
    let k_u = k[u.index()];
    let mut k_u_in: HashMap<usize, f64> = HashMap::from_iter(vec![(old_c, 0f64)].into_iter());
    for &Target { target: v, value: w} in graph.neighbors_with_values(u) {
        if v == u { continue; }
        let c = communities[v.index()];
        *k_u_in.entry(c).or_insert(0f64) += w;
    }
    let k_u_in_old_c = *k_u_in.get(&old_c).unwrap();
    k_u_in
        .into_iter()
        .map(|(c, k_u_in_c)| {
            let dq = 2. / sum_k * (k_u_in_c - k_u_in_old_c + (k_u / sum_k) * (sum_k_c[old_c] - sum_k_c[c] - k_u));
            (c, dq)
        })
        .filter(|(_, dq)| *dq > 0.)
        .max_by(|(_, dq1), (_, dq2)| dq1.total_cmp(dq2))
}


pub mod test {

    use crate::modularity::local_modularity_optimization;
    use graph_builder::{GraphBuilder, UndirectedCsrGraph};

    #[test]
    fn test_modularity() {
        let graph: UndirectedCsrGraph<usize, _, f64> = GraphBuilder::new()
            .edges_with_values(vec![(0, 1, 1.), (1, 0, 1.), (3, 4, 1.)])
            .build();
        let result = local_modularity_optimization(&graph);
        assert_eq!(result[0], result[1]);
        assert_eq!(
            result
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == result[0]).then_some(i))
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            result
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == result[2]).then_some(i))
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            result
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == result[3]).then_some(i))
                .collect::<Vec<_>>(),
            vec![3, 4]
        );
        println!("{:?}", result);
    }
}
