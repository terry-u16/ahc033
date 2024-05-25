#[derive(Debug, Clone, Copy)]
pub struct ConstVec<T: Default + Copy, const N: usize> {
    stack: [T; N],
    top: usize,
}

impl<T: Default + Copy, const N: usize> ConstVec<T, N> {
    pub fn new() -> Self {
        Self {
            stack: [T::default(); N],
            top: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.stack[self.top] = value;
        self.top += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        self.top -= 1;
        Some(self.stack[self.top])
    }

    pub fn last(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        Some(self.stack[self.top - 1])
    }

    pub fn is_empty(&self) -> bool {
        self.top == 0
    }

    pub fn len(&self) -> usize {
        self.top
    }

    pub fn as_slice(&self) -> &[T] {
        &self.stack[..self.top]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstQueue<T: Default + Copy, const N: usize> {
    queue: [T; N],
    front: usize,
}

impl<T: Default + Copy, const N: usize> ConstQueue<T, N> {
    pub fn new(values: [T; N]) -> Self {
        Self {
            queue: values,
            front: 0,
        }
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let value = self.queue[self.front];
        self.front += 1;
        Some(value)
    }

    pub fn front(&self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        Some(self.queue[self.front])
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        N - self.front
    }

    pub fn as_slice(&self) -> &[T] {
        &self.queue[self.front..]
    }
}

#[derive(Debug, Clone)]
pub struct History<T: Clone> {
    history: Vec<(T, HistoryIndex)>,
}

impl<T: Clone> History<T> {
    pub fn new() -> Self {
        History { history: vec![] }
    }

    pub fn push(&mut self, action: T, parent: HistoryIndex) -> HistoryIndex {
        let index = HistoryIndex(self.history.len());
        self.history.push((action, parent));
        index
    }

    pub fn collect(&self, mut index: HistoryIndex) -> Vec<T> {
        let mut actions = vec![];

        while index != HistoryIndex::ROOT {
            let (action, i) = self.history[index.0].clone();
            actions.push(action);
            index = i;
        }

        actions.reverse();
        actions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HistoryIndex(usize);

impl HistoryIndex {
    pub const ROOT: Self = Self(!0);
}
