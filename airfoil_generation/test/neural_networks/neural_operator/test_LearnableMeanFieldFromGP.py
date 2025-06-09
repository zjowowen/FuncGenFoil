import unittest
import torch
import gpytorch


from airfoil_generation.neural_networks.neural_operator import LearnableMeanFieldFromGP


class TestLearnableMeanFieldFromGP(unittest.TestCase):

    def test_initialization_1d(self):
        num_channels = 3
        spatial_shape = (16,)
        x_dim = 1
        model = LearnableMeanFieldFromGP(num_channels, spatial_shape, x_dim)

        self.assertEqual(model.num_channels, num_channels)
        self.assertEqual(model.spatial_shape, spatial_shape)
        self.assertEqual(model.x_dim, x_dim)
        self.assertEqual(model.num_spatial_points, 16)

        # Check the actual distribution object and its properties
        self.assertIsInstance(
            model.variational_strategy._variational_distribution,
            gpytorch.variational.MeanFieldVariationalDistribution,
        )
        self.assertEqual(
            model.variational_strategy._variational_distribution.batch_shape,
            torch.Size([num_channels]),
        )
        # Access the learnable parameter directly for shape check
        self.assertEqual(
            model.variational_strategy._variational_distribution.variational_mean.shape,
            torch.Size([num_channels, 16]),
        )
        self.assertEqual(
            model.variational_strategy.inducing_points.shape, torch.Size([16, x_dim])
        )

        self.assertIsInstance(model.mean_module, gpytorch.means.ZeroMean)
        self.assertEqual(model.mean_module.batch_shape, torch.Size([num_channels]))
        self.assertIsInstance(model.covar_module, gpytorch.kernels.ScaleKernel)
        self.assertEqual(model.covar_module.batch_shape, torch.Size([num_channels]))
        if x_dim > 0:
            self.assertIsNotNone(model.covar_module.base_kernel.ard_num_dims)
            self.assertEqual(model.covar_module.base_kernel.ard_num_dims, x_dim)

    def test_initialization_2d(self):
        num_channels = 2
        spatial_shape = (8, 8)
        x_dim = 2
        model = LearnableMeanFieldFromGP(num_channels, spatial_shape, x_dim)

        self.assertEqual(model.num_channels, num_channels)
        self.assertEqual(model.spatial_shape, spatial_shape)
        self.assertEqual(model.x_dim, x_dim)
        self.assertEqual(model.num_spatial_points, 64)

        self.assertIsInstance(
            model.variational_strategy._variational_distribution,
            gpytorch.variational.MeanFieldVariationalDistribution,
        )
        self.assertEqual(
            model.variational_strategy._variational_distribution.batch_shape,
            torch.Size([num_channels]),
        )
        # Access the learnable parameter directly for shape check
        self.assertEqual(
            model.variational_strategy._variational_distribution.variational_mean.shape,
            torch.Size([num_channels, 64]),
        )
        self.assertEqual(
            model.variational_strategy.inducing_points.shape, torch.Size([64, x_dim])
        )
        if x_dim > 0:
            self.assertEqual(model.covar_module.base_kernel.ard_num_dims, x_dim)

    def test_initialization_invalid_inputs(self):
        with self.assertRaisesRegex(TypeError, "spatial_shape must be a tuple"):
            LearnableMeanFieldFromGP(num_channels=1, spatial_shape=[16], x_dim=1)
        with self.assertRaisesRegex(
            TypeError, "All elements of spatial_shape must be positive integers"
        ):
            LearnableMeanFieldFromGP(num_channels=1, spatial_shape=(16.0,), x_dim=1)
        with self.assertRaisesRegex(
            TypeError, "All elements of spatial_shape must be positive integers"
        ):
            LearnableMeanFieldFromGP(
                num_channels=1, spatial_shape=(0,), x_dim=1
            )  # Test non-positive dim
        with self.assertRaisesRegex(
            ValueError, "Length of spatial_shape .* must match x_dim"
        ):
            LearnableMeanFieldFromGP(num_channels=1, spatial_shape=(16,), x_dim=2)
        with self.assertRaisesRegex(
            ValueError, "Length of spatial_shape .* must match x_dim"
        ):
            LearnableMeanFieldFromGP(num_channels=1, spatial_shape=(16, 16), x_dim=1)

    def test_get_learnable_mean_field(self):
        num_channels = 3
        spatial_shape = (4, 5)  # H=4, W=5
        x_dim = 2
        model = LearnableMeanFieldFromGP(num_channels, spatial_shape, x_dim)

        mean_field = model.get_learnable_mean_field()

        self.assertIsInstance(mean_field, torch.Tensor)
        self.assertEqual(mean_field.shape, torch.Size([num_channels, *spatial_shape]))
        self.assertTrue(mean_field.requires_grad)

        original_param_val = (
            model.variational_strategy._variational_distribution.variational_mean.detach().clone()
        )

        with torch.no_grad():
            mean_field[0, 0, 0] += 1.0

        self.assertEqual(
            model.variational_strategy._variational_distribution.variational_mean[
                0, 0
            ].item(),
            original_param_val[0, 0].item() + 1.0,
            "Modifying output of get_learnable_mean_field did not modify underlying variational_mean parameter.",
        )

    def test_forward_method(self):
        num_channels = 2
        spatial_shape = (8,)
        x_dim = 1
        model = LearnableMeanFieldFromGP(num_channels, spatial_shape, x_dim)
        test_x = torch.linspace(0, 1, 10).unsqueeze(-1)

        model.eval()
        output_distribution = model(test_x)

        self.assertIsInstance(
            output_distribution, gpytorch.distributions.MultivariateNormal
        )
        self.assertEqual(output_distribution.batch_shape, torch.Size([num_channels]))
        self.assertEqual(output_distribution.event_shape, torch.Size([test_x.shape[0]]))

        spatial_shape_2d = (4, 4)
        x_dim_2d = 2
        model_2d = LearnableMeanFieldFromGP(num_channels, spatial_shape_2d, x_dim_2d)
        test_x_2d = torch.rand(10, x_dim_2d)

        model_2d.eval()
        output_distribution_2d = model_2d(test_x_2d)
        self.assertIsInstance(
            output_distribution_2d, gpytorch.distributions.MultivariateNormal
        )
        self.assertEqual(output_distribution_2d.batch_shape, torch.Size([num_channels]))
        self.assertEqual(
            output_distribution_2d.event_shape, torch.Size([test_x_2d.shape[0]])
        )

    def test_learnability(self):
        num_channels = 1
        spatial_shape = (2, 2)
        x_dim = 2
        model = LearnableMeanFieldFromGP(num_channels, spatial_shape, x_dim)
        model.train()

        initial_mean_field = model.get_learnable_mean_field().clone().detach()
        optimizer_mean = torch.optim.Adam(
            [model.variational_strategy._variational_distribution.variational_mean],
            lr=0.1,
        )
        target_val_for_mean = torch.ones_like(initial_mean_field) * 5.0

        for _ in range(20):
            optimizer_mean.zero_grad()
            current_mean_param = (
                model.variational_strategy._variational_distribution.variational_mean
            )
            loss = (
                (current_mean_param.view_as(initial_mean_field) - target_val_for_mean)
                ** 2
            ).sum()
            loss.backward()
            optimizer_mean.step()

        final_mean_field = model.get_learnable_mean_field().clone().detach()
        self.assertFalse(
            torch.allclose(initial_mean_field, final_mean_field, atol=1e-3),
            f"Variational mean field did not change significantly. Diff: {(initial_mean_field - final_mean_field).abs().max()}",
        )

        if x_dim > 0 and model.covar_module.base_kernel.has_lengthscale:
            initial_lengthscale = (
                model.covar_module.base_kernel.lengthscale.clone().detach()
            )
        else:
            initial_lengthscale = None

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([num_channels])
        )
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=model.num_spatial_points
        )

        optimizer_full = torch.optim.Adam(
            [{"params": model.parameters()}, {"params": likelihood.parameters()}],
            lr=0.1,
        )

        dummy_x_for_mll = model.variational_strategy.inducing_points

        for i in range(10):
            optimizer_full.zero_grad()
            output = model(dummy_x_for_mll)
            dummy_y = torch.randn_like(output.mean) + torch.sin(torch.tensor(i / 5.0))
            loss_mll = -mll(output, dummy_y)
            loss_mll.backward()
            optimizer_full.step()

        if initial_lengthscale is not None:
            final_lengthscale = (
                model.covar_module.base_kernel.lengthscale.clone().detach()
            )
            self.assertFalse(
                torch.allclose(initial_lengthscale, final_lengthscale, atol=1e-3),
                f"Kernel lengthscale did not change. Diff: {(initial_lengthscale - final_lengthscale).abs().max()}",
            )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
