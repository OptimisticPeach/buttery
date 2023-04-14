# `buttery`
Exponentially smoothed transformations with a clean API.

## Basic Example:
```rs
use glam::{Mat4, Quat, Vec3};
use buttery::{Rotate, Scaffold, TransformComponent, Translate};
struct Camera {
    position: TransformComponent<Translate<Vec3>>,
    looking: TransformComponent<Rotate>,
}

let mut camera = Camera {
    position: TransformComponent::new_translate(Vec3::ZERO),
    looking: TransformComponent::new_rotate(Quat::IDENTITY),
};

// Simulate user input:
camera.position.target += Vec3::X;
camera.looking.target *= Quat::from_rotation_x(0.3);

// For each frame:

// Time since last frame in seconds.
let delta_time = 0.016;
let transform = camera.position.begin(|translation| Mat4::from_translation(translation))
    .and_then(&mut camera.looking, |quat| Mat4::from_quat(quat))
    .drive(delta_time);

let view_matrix = transform.inverse();
```

## Following example:
This example makes the camera follow a target.

In this case, we can't quite use the `.and_then` api:
```rs
use glam::{Mat4, Quat, Vec3};
use buttery::{Rotate, Scaffold, TransformComponent, Translate};
struct Camera {
    position: TransformComponent<Translate<Vec3>>,
    target: TransformComponent<Translate<Vec3>>,
}

let mut camera = Camera {
    position: TransformComponent::new_translate(Vec3::ZERO),
    // We'll want to follow our actual target much more closely.
    target: TransformComponent::new(0.001, Vec3::X),
};

// Simulate user input:
camera.position.target += Vec3::X;
camera.target.target += Vec3::NEG_Y;

// For each frame:

// Time since last frame in seconds.
let delta_time = 0.016;

let camera_position = camera.position.drive(delta_time);

let partial = camera.target.begin(|target_pos| Mat4::look_at_rh(camera_position, target_pos, Vec3::Y))
    .drive(delta_time);

let transform = partial * Mat4::from_translation(camera_position);

let view_matrix = transform.inverse();
```
