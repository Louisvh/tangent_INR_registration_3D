import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm

from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordtensor=None, output_shape=(28, 28), dimension=0, slice_pos=0, ret_target_mae=False, ret_fixed=False
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordtensor is None:
            coordtensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        if ret_fixed:
            input_coords_normspace = coordtensor
            fixed_image = self.transform_no_add(input_coords_normspace, self.fixed_image)
            return fixed_image.cpu().detach().numpy().reshape(output_shape[0], output_shape[1])

        coordtensor_shifted = coordtensor - self.bbox_topleft[None, :]
        coordtensor_scaled = coordtensor_shifted * self.bbox_scale[None, :]
        coordtensor_bboxspace = coordtensor_scaled - 1
        coordtensor_bboxspace[:,:2] += 0.5

        output_rel = self.network(coordtensor_bboxspace)

        coord_temp_raw = torch.add(output_rel, coordtensor_bboxspace)
        output_raw = coord_temp_raw

        output_centered = output_raw + 1
        output_centered[:,:2] -= 0.5
        output_scaled = output_centered * (1 / self.bbox_scale[None, :])
        output_shifted = output_scaled + self.bbox_topleft[None, :]

        transformed_image = self.transform_no_add(output_shifted)

        if ret_target_mae:
            fixed_image = self.transform_no_add(coordtensor, self.fixed_image)
            target_mae = torch.mean(torch.abs(fixed_image - transformed_image)).cpu().detach().numpy()
            return target_mae

        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, moving_image, fixed_image, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        for kwarg in kwargs:
            if kwarg not in self.args.keys():
                print(f'\n---\nwarning: kwarg {kwarg} not found in args!!\n---')

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.lr = kwargs["lr"] if "lr" in kwargs else self.args["lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.layers = kwargs["layers"] if "layers" in kwargs else self.args["layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.experiment_key = (
            kwargs["experiment_key"]
            if "experiment_key" in kwargs
            else self.args["experiment_key"]
        )
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        self.seed = (
            kwargs["seed"] if "seed" in kwargs else self.args["seed"]
        )
        torch.manual_seed(self.seed)

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )
        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.network = networks.MLP(self.layers)
            else:
                self.network = networks.Siren(self.layers, self.weight_init, self.omega)
            if self.verbose:
                print(
                    "Network contains {} trainable parameters.".format(
                        general.count_parameters(self.network)
                    )
                )
        else:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()

        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Choose the optimizer
        m_params = list(self.network.parameters())
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                m_params, lr=self.lr, momentum=self.momentum
            )
        elif self.optimizer_arg.lower() == "adamw":
            self.optimizer = optim.AdamW(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(m_params, lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(m_params, lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                m_params, lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()

        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Parse arguments from kwargs
        self.mask = kwargs["mask"] if "mask" in kwargs else self.args["mask"]

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image

        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(
            self.mask, self.fixed_image.shape
        )

        self.bbox_topleft = torch.stack((torch.min(self.possible_coordinate_tensor[:,0]),
                                        torch.min(self.possible_coordinate_tensor[:,1]),
                                        torch.min(self.possible_coordinate_tensor[:,2])))

        self.bbox_botright = torch.stack((torch.max(self.possible_coordinate_tensor[:,0]),
                                        torch.max(self.possible_coordinate_tensor[:,1]),
                                        torch.max(self.possible_coordinate_tensor[:,2])))

        self.bbox_scale = 1 / ((self.bbox_botright - self.bbox_topleft) / 2)

        # match coordinate space of mpr model
        self.bbox_scale[:2] *= 0.5


        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()

    def cuda(self):
        """Move the model to the GPU."""

        # Standard variables
        self.network.cuda()

        # Variables specific to this class
        self.moving_image.cuda()
        self.fixed_image.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None

        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["batch_size"] = 10000
        self.args["layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "saved_models"
        self.args["experiment_key"] = "default"

        self.args["network_type"] = "Siren"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32

        self.args["seed"] = 1

        # suppress warnings when comparing with tangent model
        unused_args = ['manifold_diam', 'voxel_spacing']
        for marg in unused_args:
            self.args[marg] = None

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device="cuda"
        )[: self.batch_size]
        coordtensor_raw = self.possible_coordinate_tensor[indices, :]
        coordtensor_shifted = coordtensor_raw - self.bbox_topleft[None, :]
        coordtensor_scaled = coordtensor_shifted * self.bbox_scale[None, :]
        coordtensor = coordtensor_scaled - 1
        coordtensor[:,:2] += 0.5
        coordtensor = coordtensor.requires_grad_(True)
        gridcoords = coordtensor_raw.reshape((1,1,1,coordtensor_raw.shape[0],3)).flip(-1)
        fixed_coord_samples = torch.nn.functional.grid_sample(
                                self.fixed_image[None, None, :], gridcoords,
                                align_corners=True).flatten()
        del coordtensor_raw, coordtensor_shifted, coordtensor_scaled

        forward_est_rel = self.network(coordtensor)
        forward_est_bboxspace = torch.add(forward_est_rel, coordtensor)
        forward_est_centered = forward_est_bboxspace + 1
        forward_est_centered[:,:2] -= 0.5
        forward_est_scaled = forward_est_centered * (1/self.bbox_scale[None, :])
        rescoords_fw = forward_est_scaled + self.bbox_topleft[None, :]
        gridcoords_fw = rescoords_fw.reshape((1,1,1,rescoords_fw.shape[0],3)).flip(-1)
        transformed_samples_fw = torch.nn.functional.grid_sample(
                                self.moving_image[None, None, :], gridcoords_fw,
                                align_corners=True).flatten()
        del forward_est_centered, forward_est_scaled

        # Compute the loss
        loss += self.criterion(transformed_samples_fw, fixed_coord_samples)

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy()

        # Regularization
        if self.jacobian_regularization:
            loss += self.alpha_jacobian * regularizers.compute_jacobian_loss(
                coordtensor, forward_est_rel, batch_size=self.batch_size
            )
        if self.hyper_regularization:
            loss += self.alpha_hyper * regularizers.compute_hyper_elastic_loss(
                coordtensor, forward_est_rel, batch_size=self.batch_size
            )
        if self.bending_regularization:
            loss += self.alpha_bending * regularizers.compute_bending_energy(
                coordtensor, forward_est_rel, batch_size=self.batch_size
            )

        # Perform the backpropagation and update the parameters accordingly
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def transform(
        self, transformation, coordtensor=None, moving_image=None, reshape=False
    ):
        """Transform moving image given a transformation."""

        # If no specific coordinate tensor is given use the standard one of 28x28
        if coordtensor is None:
            coordtensor = self.coordtensor

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        # From relative to absolute
        transformation = torch.add(transformation, coordtensor)
        return general.fast_trilinear_interpolation(
            moving_image,
            transformation[:, 0],
            transformation[:, 1],
            transformation[:, 2],
        )

    def transform_no_add(self, transformation, moving_image=None):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image

        gridcoords = transformation.reshape((1, 1, 1, transformation.shape[0], 3)).flip(-1)
        fixed_image = torch.nn.functional.grid_sample(moving_image[None, None, :],
                gridcoords, align_corners=True).flatten()
        return fixed_image

    def fit(self, epochs=None, red_blue=False):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.seed)

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i)
            if i % (epochs//50) == 0 and self.verbose:
                self.save_losslogs()

        self.savenets()
        if self.verbose:
            print('loss (start, middle, end)')
            print(self.loss_list[0], self.loss_list[epochs//2], self.loss_list[-1])
            self.save_losslogs()

    def save_losslogs(self):
        import numpy as np
        np.savetxt(f'{self.save_folder}/loss_log_{self.experiment_key}.txt',
                self.loss_list)
        np.savetxt(f'{self.save_folder}/data_loss_log_{self.experiment_key}.txt',
                self.data_loss_list)

    def savenets(self):
        f_fname = f'{self.save_folder}/F_{self.experiment_key}.pth'
        torch.save(self.network.state_dict(), f_fname)

