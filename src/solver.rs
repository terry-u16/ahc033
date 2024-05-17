use crate::{
    common::ChangeMinMax,
    grid::Coord,
    problem::{Input, Operation, Output, Yard},
};

pub fn solve(input: &Input) -> Result<Output, ()> {
    let mut yard = Yard::new(&input);
    let mut output = Output::new();

    // 取り出す
    let rule = vec![
        Operation::Pick,
        Operation::Right,
        Operation::Right,
        Operation::Right,
        Operation::Drop,
        Operation::Left,
        Operation::Left,
        Operation::Left,
        Operation::Pick,
        Operation::Right,
        Operation::Right,
        Operation::Drop,
        Operation::Left,
        Operation::Left,
        Operation::Pick,
        Operation::Right,
        Operation::Drop,
    ];

    for &operation in rule.iter() {
        let operations = [operation; Input::N];
        yard.apply(&operations)?;
        yard.carry_in_and_ship();
        output.push(&operations);
    }

    // 爆破
    let operation = [
        Operation::None,
        Operation::Destroy,
        Operation::Destroy,
        Operation::Destroy,
        Operation::Destroy,
    ];
    yard.apply(&operation)?;
    output.push(&operation);

    // 運ぶ
    while !yard.is_end() {
        if let Some(next_c) = find_next_pick(&yard) {
            let container = yard.grid()[next_c].unwrap();
            move_to(&mut yard, &mut output, next_c)?;
            apply_single(&mut yard, &mut output, Operation::Pick)?;
            move_to(&mut yard, &mut output, Input::get_goal(container))?;
            apply_single(&mut yard, &mut output, Operation::Drop)?;
        } else if let Some((start, goal)) = find_next_stock(&yard) {
            move_to(&mut yard, &mut output, start)?;
            apply_single(&mut yard, &mut output, Operation::Pick)?;
            move_to(&mut yard, &mut output, goal)?;
            apply_single(&mut yard, &mut output, Operation::Drop)?;
        } else {
            let c = yard.cranes()[0].coord().unwrap();
            let container = yard.grid()[c].unwrap();
            apply_single(&mut yard, &mut output, Operation::Pick)?;
            move_to(&mut yard, &mut output, Input::get_goal(container))?;
            apply_single(&mut yard, &mut output, Operation::Drop)?;
        }
    }

    Ok(output)
}

fn move_to(yard: &mut Yard, output: &mut Output, destination: Coord) -> Result<(), ()> {
    while yard.cranes()[0].coord().unwrap().row() < destination.row() {
        apply_single(yard, output, Operation::Down)?;
    }

    while yard.cranes()[0].coord().unwrap().row() > destination.row() {
        apply_single(yard, output, Operation::Up)?;
    }

    while yard.cranes()[0].coord().unwrap().col() < destination.col() {
        apply_single(yard, output, Operation::Right)?;
    }

    while yard.cranes()[0].coord().unwrap().col() > destination.col() {
        apply_single(yard, output, Operation::Left)?;
    }

    Ok(())
}

fn apply_single(yard: &mut Yard, output: &mut Output, operation: Operation) -> Result<(), ()> {
    let mut operations = [Operation::None; Input::N];

    operations[0] = operation;
    yard.apply(&operations)?;
    output.push(&operations);
    yard.carry_in_and_ship();

    Ok(())
}

fn find_next_pick(yard: &Yard) -> Option<Coord> {
    for row in 0..Input::N {
        for col in 0..Input::N - 1 {
            let c = Coord::new(row, col);
            let Some(container) = yard.grid()[c] else {
                continue;
            };

            if yard.is_next_ship(container) {
                return Some(c);
            }
        }
    }

    None
}

fn find_next_stock(yard: &Yard) -> Option<(Coord, Coord)> {
    let mut best_pair = None;
    let mut best_cost = usize::MAX;

    for start_row in 0..Input::N {
        let start = Coord::new(start_row, 0);

        if yard.grid()[start].is_none() || yard.waiting()[start_row].is_empty() {
            continue;
        }

        for row in 0..Input::N {
            for col in 0..Input::N - 1 {
                let goal = Coord::new(row, col);

                if yard.grid()[goal].is_none() {
                    let cost = yard.cranes()[0].coord().unwrap().dist(&start) + start.dist(&goal);

                    if best_cost.change_min(cost) {
                        best_pair = Some((start, goal));
                    }
                }
            }
        }
    }

    best_pair
}
