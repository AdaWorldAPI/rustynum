pub mod dimension;
pub mod storage;
pub mod order;
pub mod array;
pub mod aliases;

pub use dimension::{Axis, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, IntoDimension};
pub use storage::{OwnedStorage, RawStorage, Storage, StorageMut, ViewStorage, ViewMutStorage};
pub use order::{Order, LayoutFlags};
pub use array::NdArray;
pub use aliases::*;
