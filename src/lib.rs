//!
//! # `buttery`
//! Exponentially smoothed transformations with a clean API.
//!
//! ## Basic Example:
//! ```
//! use glam::{Mat4, Quat, Vec3};
//! use buttery::{Rotate, Scaffold, TransformComponent, Translate};
//! struct Camera {
//!     position: TransformComponent<Translate<Vec3>>,
//!     looking: TransformComponent<Rotate>,
//! }
//!
//! let mut camera = Camera {
//!     position: TransformComponent::new_translate(Vec3::ZERO),
//!     looking: TransformComponent::new_rotate(Quat::IDENTITY),
//! };
//!
//! // Simulate user input:
//! camera.position.target += Vec3::X;
//! camera.looking.target *= Quat::from_rotation_x(0.3);
//!
//! // For each frame:
//!
//! // Time since last frame in seconds.
//! let delta_time = 0.016;
//! let transform = camera.position.begin(|translation| Mat4::from_translation(translation))
//!     .and_then(&mut camera.looking, |quat| Mat4::from_quat(quat))
//!     .drive(delta_time);
//!
//! let view_matrix = transform.inverse();
//! ```
//!
//! ## Following example:
//! This example makes the camera follow a target.
//!
//! In this case, we can't quite use the `.and_then` api:
//! ```
//! use glam::{Mat4, Quat, Vec3};
//! use buttery::{Rotate, Scaffold, TransformComponent, Translate};
//! struct Camera {
//!     position: TransformComponent<Translate<Vec3>>,
//!     target: TransformComponent<Translate<Vec3>>,
//! }
//!
//! let mut camera = Camera {
//!     position: TransformComponent::new_translate(Vec3::ZERO),
//!     // We'll want to follow our actual target much more closely.
//!     target: TransformComponent::new(0.001, Vec3::X),
//! };
//!
//! // Simulate user input:
//! camera.position.target += Vec3::X;
//! camera.target.target += Vec3::NEG_Y;
//!
//! // For each frame:
//!
//! // Time since last frame in seconds.
//! let delta_time = 0.016;
//!
//! let camera_position = camera.position.drive(delta_time);
//!
//! let partial = camera.target.begin(|target_pos| Mat4::look_at_rh(camera_position, target_pos, Vec3::Y))
//!     .drive(delta_time);
//!
//! let transform = partial * Mat4::from_translation(camera_position);
//!
//! let view_matrix = transform.inverse();
//! ```

use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
use glam::{Mat4, Quat};

/// Describes a smoothed attribute, such as rotation or translation.
pub trait Smoothed {
    /// The actual type that represents this attribute.
    type Attribute: Copy;
    /// "Drives" the current value towards the target by the percent.
    /// Expected to be some kind of linear interpolation.
    fn drive(target: Self::Attribute, current: Self::Attribute, percent: f32) -> Self::Attribute;
}

/// Describes the current state of a smoothed attribute.
#[derive(Copy, Clone, Debug)]
pub struct TransformComponent<T: Smoothed> {
    /// How close should the current value follow the target.
    ///
    /// Default value depends on the `new_` function you choose,
    /// however reasonable values are near `0.05` or so. Closer to
    /// `0.0` yields closer following, closer to `1.0` yields
    /// more lenient following.
    pub retention: f32,
    /// The current value.
    pub current: T::Attribute,
    /// The target value.
    pub target: T::Attribute,
    _unused: PhantomData<T>,
}

impl<T: Smoothed> TransformComponent<T> {
    /// Drives the attribute forward using exponential smoothing by
    /// `delta_time` seconds since the last update.
    ///
    /// This usually isn't called manually, and instead the [`begin`](Self::begin) interface is preferred.
    pub fn drive(&mut self, delta_time: f32) -> T::Attribute {
        let percent = 1.0 - self.retention.powf(delta_time);
        let new_current = T::drive(self.target, self.current, percent);
        self.current = new_current;
        new_current
    }

    /// Forcibly sets the target and current value to something.
    ///
    /// This snaps the values to the requested target.
    pub fn hard_set(&mut self, target: T::Attribute) {
        self.target = target;
        self.current = target;
    }

    /// Creates a new `TransformComponent` with the requested
    /// retention and initial value. See the field documentation
    /// for details on how `retention` works.
    pub fn new(retention: f32, initial: T::Attribute) -> Self {
        Self {
            retention,
            current: initial,
            target: initial,
            _unused: PhantomData,
        }
    }

    /// Begins a transformation.
    ///
    /// The function parameter `f` translates the actual value into
    /// a matrix so that compositions of transforms may be taken.
    ///
    /// ## Ordering
    /// If you have:
    /// ```
    /// # use glam::{Mat4, Vec3};
    /// # use buttery::{TransformComponent, Scaffold};
    /// let mut a = TransformComponent::new_zoom(3.0);
    /// let mut b = TransformComponent::new_translate(Vec3::new(1.0, 3.0, 5.0));
    /// let transform_matrix = a.begin(|zoom| Mat4::from_scale(Vec3::ONE * zoom))
    ///     .and_then(&mut b, |translation| Mat4::from_translation(translation))
    ///     .drive(0.016);
    /// ```
    /// Then `transform_matrix` is the resulting matrix of first applying the
    /// scale and then the translation. Namely the final matrix would be equivalent to:
    /// ```
    /// # use glam::{Mat4, Vec3};
    /// let zoom = 3.0;
    /// let translation = Vec3::new(1.0, 3.0, 5.0);
    /// let transform_matrix = Mat4::from_translation(translation) * Mat4::from_scale(Vec3::ONE * zoom);
    /// ```
    pub fn begin<F: FnOnce(T::Attribute) -> Mat4>(&mut self, f: F) -> First<T, F> {
        First {
            component: self,
            f,
        }
    }
}

impl<T> TransformComponent<Translate<T>>
where T: Add<T, Output=T> + Mul<f32, Output=T> + Sub<T, Output=T> + Copy
{
    /// Creates a new `TransformComponent` with a retention of `0.01`.
    pub fn new_translate(initial_state: T) -> Self {
        Self::new(0.01, initial_state)
    }

    /// Creates a new `TransformComponent` with a retention of `0.03`.
    pub fn new_zoom(initial_state: T) -> Self {
        Self::new(0.03, initial_state)
    }

    /// Creates a new `TransformComponent` with a retention of `0.04`.
    pub fn new_angle(initial_state: T) -> Self {
        Self::new(0.04, initial_state)
    }
}

impl TransformComponent<Rotate> {
    /// Creates a new `TransformComponent<Rotate>` with a retention of `0.04`.
    pub fn new_rotate(initial_state: Quat) -> Self {
        Self::new(0.04, initial_state)
    }
}

/// Represents anything whose interpolation looks like `(1 - t) * a + t * (b - a)`.
pub struct Translate<T>(PhantomData<T>);

impl<T> Smoothed for Translate<T>
where T: Add<T, Output = T> + Mul<f32, Output = T> + Sub<T, Output = T> + Copy {
    type Attribute = T;
    fn drive(target: T, current: T, percent: f32) -> T {
        current + (target - current) * percent
    }
}

/// Represents quaternion interpolation through [`slerp`](Quat::slerp).
pub struct Rotate;

impl Smoothed for Rotate {
    type Attribute = Quat;
    fn drive(target: Quat, current: Quat, percent: f32) -> Quat {
        current.slerp(target, percent).normalize()
    }
}

/// Implementation detail. Yielded from [`.begin`](TransformComponent::begin).
pub struct First<'a, T: Smoothed, F: FnOnce(T::Attribute) -> Mat4> {
    component: &'a mut TransformComponent<T>,
    f: F,
}

/// Result of calling [`.and_then`](Scaffold::and_then).
pub struct Composition<'a, T: Smoothed, F: FnOnce(T::Attribute) -> Mat4, I: Scaffold + 'a> {
    component: &'a mut TransformComponent<T>,
    f: F,
    inner: I,
}

/// Represents a transform that can be proceeded by another one.
pub trait Scaffold: Sized {
    /// Finishes the current series of transformations.
    fn drive(self, time: f32) -> Mat4;

    /// Queues another transformation to happen after the previous one(s).
    #[inline(always)]
    fn and_then<'a, T: Smoothed, F: FnOnce(T::Attribute) -> Mat4>(self, next: &'a mut TransformComponent<T>, f: F) -> Composition<'a, T, F, Self>
        where Self: 'a {
        Composition {
            component: next,
            f,
            inner: self,
        }
    }
}

impl<'a, T: Smoothed, F: FnOnce(T::Attribute) -> Mat4> Scaffold for First<'a, T, F> {
    #[inline(always)]
    fn drive(self, time: f32) -> Mat4 {
        let attrib = self.component.drive(time);
        (self.f)(attrib)
    }
}

impl<'a, T, F, I> Scaffold for Composition<'a, T, F, I>
where T: Smoothed,
    F: FnOnce(T::Attribute) -> Mat4,
    I: Scaffold + 'a {
    #[inline(always)]
    fn drive(self, time: f32) -> Mat4 {
        let inner = self.inner.drive(time);
        let attrib = self.component.drive(time);
        (self.f)(attrib) * inner
    }
}

#[cfg(test)]
mod test {
    use glam::Vec3;
    use super::*;

    #[test]
    fn this_works() {
        let mut zoom = TransformComponent::new_zoom(2.0);
        let mut rotate = TransformComponent::new_rotate(Quat::IDENTITY);
        let mut translate = TransformComponent::new_translate(Vec3::ONE);
        let mut angle = TransformComponent::new_angle(0.4);

        zoom.target = 4.0;
        rotate.target *= Quat::from_rotation_x(0.4);
        translate.target += Vec3::ONE;
        angle.target /= 3.0;

        let delta_time = 0.03;

        let transform_matrix = zoom.begin(|zoom| Mat4::from_translation(Vec3::splat(zoom)))
            .and_then(&mut rotate, |quat| Mat4::from_quat(quat))
            .and_then(&mut translate, |by| Mat4::from_translation(by))
            .and_then(&mut angle, |angle| Mat4::from_rotation_y(angle))
            .drive(delta_time);

        let inv = transform_matrix.inverse();
        assert!(inv.is_finite());
    }
}
