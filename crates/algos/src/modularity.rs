use graph_builder::prelude::*;
use std::collections::{HashMap, HashSet};
use rand::prelude::SliceRandom;
use rand::thread_rng;

pub fn local_modularity_optimization<NI, G>(graph: &G, max_iterations: u32, improvement_threshold: f64) -> (Vec<usize>, f64)
where
    NI: Idx,
    G: Graph<NI> + UndirectedNeighborsWithValues<NI, f64>,
{
    let node_count = graph.node_count().index();
    let k = (0..node_count) //can be parallelized
        .map(|u| {
            graph
                .neighbors_with_values(NI::new(u))
                .map(|Target { target: _, value }| value)
                .sum()
        })
        .collect::<Vec<_>>();
    let mut sigma = k.clone();
    let inv_2m = 1f64 / k.iter().sum::<f64>();

    let mut communities: Vec<_> = (0..node_count).collect();

    let mut q = modularity(graph, &communities);
    for i in 0..max_iterations {
        let mut total_diff = 0.;
        for u in 0..node_count {
            let old_c = communities[u];
            match best_community(
                graph,
                &communities,
                &sigma,
                &k,
                inv_2m,
                NI::new(u)
            ) {
                Some((new_c, diff)) => {
                    total_diff += diff;
                    communities[u] = new_c;
                    sigma[new_c] += k[u];
                    sigma[old_c] -= k[u];
                },
                None => {}
            }
        }
        // let q = modularity(graph, &communities);
        q += total_diff;
        println!("Iterations: {}, Modularity: {}, Diffs: {}", i, q, total_diff);
        if total_diff < improvement_threshold {
            return (communities, q)
        }
    }
    (communities, q)
}

pub fn modularity<NI, G>(graph: &G, communities: &Vec<usize>) -> f64
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
    sigma: &Vec<f64>,
    k: &Vec<f64>,
    inv_2m: f64,
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
        let c = communities[v.index()];
        *k_u_in.entry(c).or_insert(0f64) += w;
    }
    let k_u_in_old_c = *k_u_in.get(&old_c).unwrap();
    k_u_in
        .into_iter()
        .map(|(c, k_u_in_c)| {
            let dq = 2. * inv_2m * (k_u_in_c - k_u_in_old_c + k_u * inv_2m * (sigma[old_c] - sigma[c] - k_u));
            (c, dq)
        })
        .filter(|(_, dq)| *dq > 0.)
        .max_by(|(_, dq1), (_, dq2)| dq1.total_cmp(dq2))
}


pub mod test {
    use crate::modularity::local_modularity_optimization;
    use graph_builder::{CsrLayout, GraphBuilder, UndirectedCsrGraph};

    #[test]
    fn test_modularity() {
        let graph: UndirectedCsrGraph<usize, _, f64> = GraphBuilder::new()
            .csr_layout(CsrLayout::Unsorted)
            .edges_with_values(vec![(0, 1, 1.), (1, 0, 1.), (3, 4, 1.)])
            .build();
        let (communities, q) = local_modularity_optimization(&graph, 10, 0.);
        assert_eq!(communities[0], communities[1]);
        assert_eq!(
            communities
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == communities[0]).then_some(i))
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            communities
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == communities[2]).then_some(i))
                .collect::<Vec<_>>(),
            vec![2]
        );
        assert_eq!(
            communities
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (*c == communities[3]).then_some(i))
                .collect::<Vec<_>>(),
            vec![3, 4]
        );
        println!("{:?}", communities);
    }
}
