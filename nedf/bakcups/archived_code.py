# # Conversion function
# def convert_old_to_new_format(data, near_thresh=0.01, far_thresh=500.0):
#     # Extract camera extrinsics, focal length, and depth map
#     testpose = torch.from_numpy(data["pose"]).float().to(device)
#     focal_length = torch.from_numpy(data["focal"]).float().to(device)
#     testimg = torch.from_numpy(data["depth_map"]).float().to(device)

#     # Height and width of the depth map
#     height, width = testimg.shape[:2]

#     # Generate ray bundle
#     ray_origins, ray_directions = get_ray_bundle(height, width, focal_length, testpose)

#     # Package into the dataset format
#     new_data_format = {
#         0: {  # Use a dummy index 0 for a single item
#             RAYS_ENTRY: {
#                 RAY_ORIGINS_ENTRY: ray_origins,
#                 RAY_DIRECTIONS_ENTRY: ray_directions
#             },
#             DEPTH_MAP_ENTRY: testimg,
#             CAMERA_POS_ENTRY: testpose[:3, 3],  # Extract camera position from pose
#             CAMERA_ANGLE_ENTRY: testpose[:3, :3],  # Extract rotation matrix
#         }
#     }

#     print("Successfully converted old data format to new format.")
#     return new_data_format

# # Load the old data
# old_data = np.load("depth_map_test.npz")

# # Convert it to the new format
# converted_dataset = convert_old_to_new_format(old_data)

# # Use the converted data in the training pipeline
# test_dataset = converted_dataset
# DATASET_SIZE = len(test_dataset)
# print(f"Loaded dataset with {DATASET_SIZE} entries!")

# resized_depth_map = cv2.resize(
#     dataset[0][DEPTH_MAP_ENTRY], 
#     (100, 100),
#     interpolation=cv2.INTER_LINEAR
# )

# resized_ray_origins = cv2.resize(
#     dataset[0][RAYS_ENTRY][RAY_ORIGINS_ENTRY].clone().cpu().numpy(), 
#     (100, 100),
#     interpolation=cv2.INTER_LINEAR
# )

# resized_ray_directions = cv2.resize(
#     dataset[0][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY].clone().cpu().numpy(), 
#     (100, 100),
#     interpolation=cv2.INTER_LINEAR
# )

# test_dataset[0][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY] = torch.from_numpy(resized_ray_directions).to(device)
# test_dataset[0][RAYS_ENTRY][RAY_ORIGINS_ENTRY] = torch.from_numpy(resized_ray_origins).to(device)
# print(test_dataset[0][RAYS_ENTRY][RAY_ORIGINS_ENTRY][0, 0])
# test_dataset[0][DEPTH_MAP_ENTRY] = resized_depth_map

# # Plot the resized depth map
# plt.imshow(test_dataset[0][DEPTH_MAP_ENTRY], cmap='inferno')
# plt.colorbar()
# plt.title("New Depth Map (100x100)")
# plt.show()

# plt.imshow(test_dataset[0][RAYS_ENTRY][RAY_ORIGINS_ENTRY].cpu().numpy())
# plt.colorbar()
# plt.title("New origins Map (100x100)")
# plt.show()

# plt.imshow(test_dataset[0][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY].cpu().numpy())
# plt.colorbar()
# plt.title("New directions Map (100x100)")
# plt.show()

# import os
# import torch
# import random

# # Training parameters
# L = 5
# lr = 1e-4
# num_iters = 500000
# display_every = 10
# save_every = 50  # Save every n iterations
# checkpoint_dir = "checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)

# # Model and optimizer
# model = VeryTinyNeDFModel(L=L).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # Seed RNG
# seed = 9458
# torch.manual_seed(seed)
# np.random.seed(seed)

# # Hyperparameters
# eikonal_penalty_weight = 5
# batch_size = 1024  # Number of rays per batch

# # Load previous checkpoint if available
# # start_epoch, best_loss = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, "latest_checkpoint.pth"))
# start_epoch = 0
# best_loss = 1e10

# # Training Loop (Random Ray Sampling Without Loop)
# for i in range(start_epoch, num_iters):
#     # Extract data for all sampled indices
#     data = list(test_dataset.values())[0]


#     # Gather depths to filter out pixels with depth < 50

#     target_depth = torch.tensor(data[DEPTH_MAP_ENTRY], dtype=torch.float32).to(device)
#     # Add batch and channel dimensions for interpolation

#     min_depth = target_depth.min().item()
#     max_depth = target_depth.max().item()
#     avg_depth = target_depth.mean().item()

#     # print(f"min depth: {min_depth}")
#     # print(f"max depth: {max_depth}")
#     # print(f"avg depth: {avg_depth}")

#     # Filter data using the mask
#     ray_origins = torch.tensor(data[RAYS_ENTRY][RAY_ORIGINS_ENTRY], dtype=torch.float32).to(device)
#     ray_directions = torch.tensor(data[RAYS_ENTRY][RAY_DIRECTIONS_ENTRY], dtype=torch.float32).to(device)

#     # Perform sphere tracing
#     depth_predicted, steps, query_points, query_results = render_depth_sphere_tracing(
#         model, ray_origins, ray_directions, target_depth, near_thresh
#     )

#     # Compute Loss
#     gradients = compute_gradients(model, query_points)
#     eikonal_loss = compute_eikonal_loss(gradients)
#     depth_loss = torch.nn.functional.mse_loss(depth_predicted, target_depth)
#     total_loss = depth_loss + eikonal_loss * eikonal_penalty_weight

#     # Backpropagation
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     # Display Progress
#     if i % display_every == 0:
#         print(f"\n### Step {i}: ###")
#         print(f"Average steps: {steps.mean().item():.2f}")
#         print(f"Total Loss: {total_loss.item():.4f}")
#         print(f"Depth Loss: {depth_loss.item():.4f}")
#         print(f"Eikonal Loss: {eikonal_loss.item():.4f}")

#     # Save checkpoint periodically or if loss improves
#     # if i % save_every == 0 or total_loss.item() < best_loss:
#     #     best_loss = total_loss.item()
#     #     save_checkpoint(model, optimizer, i, total_loss.item(), os.path.join(checkpoint_dir, "latest_checkpoint.pth"))

# print("Training complete!")