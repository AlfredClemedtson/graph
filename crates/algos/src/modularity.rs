use graph_builder::prelude::*;
use std::collections::{HashMap, HashSet};
use std::mem::replace;

const IMPROVEMENT_THRESHOLD: f64 = 0.0001;

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
    let mut communities = (0..node_count).collect();
    let mut counter = 0;
    loop {
        let mut total_diff = 0f64;
        for u in 0..node_count {
            //process nodes of the same color in parallel
            match best_community(
                graph,
                &communities,
                &weighted_degree_of_community,
                &weighted_degree,
                NI::new(u),
            ) {
                Some((new_c, diff)) => {
                    let old_c = replace(&mut communities[u], new_c);
                    weighted_degree_of_community[new_c] += weighted_degree[u];
                    weighted_degree_of_community[old_c] -= weighted_degree[u];
                    total_diff += diff;
                }
                None => {}
            }
        }
        if total_diff > IMPROVEMENT_THRESHOLD {
            counter += 1;
            continue;
        }
        let q = modularity(graph, &communities);
        println!("Iterations: {}, Modularity: {}", counter, q);
        return communities;
    }
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
    let edge_count = graph.edge_count().index() as f64;
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
            sum_in_c / (2. * edge_count) - (sum_tot / (2. * edge_count)).powi(2)
        })
        .sum()
}
fn best_community<NI, G>(
    graph: &G,
    communities: &Vec<usize>,
    tot: &Vec<f64>,
    wdeg: &Vec<f64>,
    u: NI,
) -> Option<(usize, f64)>
where
    NI: Idx,
    G: Graph<NI> + UndirectedNeighborsWithValues<NI, f64>,
{
    let one_over_2m = 1. / graph.edge_count().index() as f64; //compute once instead

    let mut ku_in = HashMap::new();
    for &Target {
        target: v,
        value: weight,
    } in graph.neighbors_with_values(u)
    {
        let c = communities[v.index()];
        *ku_in.entry(c).or_insert(0f64) += weight;
    }
    let old_c = communities[u.index()];
    let ku_in_c = *ku_in.get(&old_c).unwrap_or(&0f64);
    let old_dq = delta_q(one_over_2m, ku_in_c, tot[old_c], wdeg[u.index()]);
    ku_in
        .into_iter()
        .map(|(c, ku_in_c)| (c, delta_q(one_over_2m, ku_in_c, tot[c], wdeg[u.index()])))
        .filter(|&(_, dq): &(usize, f64)| dq > old_dq)
        .max_by(|(_, dq1), (_, dq2)| dq1.total_cmp(dq2))
        .map(|(c, dq)| (c, dq - old_dq))
}

fn delta_q(one_over_2m: f64, ku_in_c: f64, tot_c: f64, wdeg_u: f64) -> f64 {
    2. * one_over_2m * (ku_in_c - one_over_2m * tot_c * wdeg_u)
}

pub mod test {

    use graph_builder::{GraphBuilder, UndirectedCsrGraph};
    use crate::modularity::local_modularity_optimization;

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
            vec![3,4]
        );
        println!("{:?}", result);
    }
}
