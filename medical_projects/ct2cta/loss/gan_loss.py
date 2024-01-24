# !/usr/bin/env python3
# coding=utf-8
import torch as th
import torch.nn.functional as F


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss(object):
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self):
        """
        init
        """
        pass

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        Args:
            real_samps:
            fake_samps:
            netD:
        Returns:
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, netD):
        """
        Args:
            real_samps:
            fake_samps:
            netD:
        Returns:
        """
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):
    """
    StandardGAN
    """
    def __init__(self):
        """
        init
        """
        from torch.nn import BCEWithLogitsLoss

        super(StandardGAN, self).__init__()

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        Args:
            real_samps:
            fake_samps:
            netD:
        Returns:
        """
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = netD(real_samps)
        f_preds = netD(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            th.squeeze(r_preds),
            th.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            th.squeeze(f_preds),
            th.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, netD):
        """
        Args:
            _:
            fake_samps:
            netD:
        Returns:
        """
        preds, _, _ = netD(fake_samps)
        return self.criterion(th.squeeze(preds),
                              th.ones(fake_samps.shape[0]).to(fake_samps.device))


class WGANGP(GANLoss):
    """
    WGAN_GP
    """
    def __init__(self, drift=0.001, use_gp=False):
        """
        Args:
            drift:
            use_gp:
        """
        super(WGANGP, self).__init__()
        self.drift = drift
        self.use_gp = use_gp

    def __gradient_penalty(self, real_samps, fake_samps, netD, reg_lambda=10):
        """
        Args:
            real_samps:
            fake_samps:
            netD:
            reg_lambda:
        Returns:
        """
        batch_size = real_samps.shape[0]

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = (epsilon * real_samps) + ((1 - epsilon) * fake_samps)
        merged.requires_grad = True

        # forward pass
        op = netD(merged)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        # define the (Wasserstein) loss
        fake_out = netD(fake_samps)
        real_out = netD(real_samps)

        loss = (th.mean(fake_out) - th.mean(real_out)
                + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, netD)
            loss += gp

        return loss

    def gen_loss(self, _, fake_samps, netD):
        """
        :param _:
        :param fake_samps:
        :return:
        """
        # calculate the WGAN loss for generator
        loss = -th.mean(netD(fake_samps))

        return loss


class LSGAN(GANLoss):
    """
    LSGAN
    """
    def __init__(self):
        """
        :param dis:
        """
        super(LSGAN, self).__init__()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        # return 0.5 * (((th.mean(self.dis(real_samps)) - 1) ** 2) + (th.mean(self.dis(fake_samps))) ** 2)
        return 1.0 * (((th.mean(netD(real_samps)) - 1) ** 2) + (th.mean(netD(fake_samps))) ** 2)

    def gen_loss(self, _, fake_samps, netD):
        """
        :param _:
        :param fake_samps:
        :return:
        """
        # return 0.5 * ((th.mean(self.dis(fake_samps)) - 1) ** 2)
        return 1.0 * ((th.mean(netD(fake_samps)) - 1) ** 2)


class LSGANSIGMOID(GANLoss):
    """
    LSGAN_SIGMOID
    """
    def __init__(self):
        """
        init
        """
        super(LSGANSIGMOID, self).__init__()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        from torch.nn.functional import sigmoid
        real_scores = th.mean(sigmoid(netD(real_samps)))
        fake_scores = th.mean(sigmoid(netD(fake_samps)))
        return 0.5 * (((real_scores - 1) ** 2) + (fake_scores ** 2))

    def gen_loss(self, _, fake_samps, netD):
        """
        :param _:
        :param fake_samps:
        :return:
        """
        from torch.nn.functional import sigmoid
        scores = th.mean(sigmoid(netD(fake_samps)))
        return 0.5 * ((scores - 1) ** 2)


class HingeGAN(GANLoss):
    """
    HingeGAN
    """
    def __init__(self):
        """
        init
        """
        super(HingeGAN, self).__init__()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        r_preds, r_mus, r_sigmas = netD(real_samps)
        f_preds, f_mus, f_sigmas = netD(fake_samps)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, netD):
        """
        :param _:
        :param fake_samps:
        :return:
        """
        return -th.mean(netD(fake_samps))


class RelativisticAverageHingeGAN(GANLoss):
    """
    RelativisticAverageHingeGAN
    """
    def __init__(self):
        """
        init
        """
        super(RelativisticAverageHingeGAN, self).__init__()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        # Obtain predictions
        r_preds = netD(real_samps)
        f_preds = netD(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        # Obtain predictions
        r_preds = netD(real_samps)
        f_preds = netD(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))


class NonSaturatingGAN(GANLoss):
    """
    NonSaturatingGAN
    """
    def __init__(self):
        """
        init
        """
        super(NonSaturatingGAN, self).__init__()

    def dis_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        # Obtain predictions
        r_preds = netD(real_samps)
        f_preds = netD(fake_samps)

        real_loss = F.softplus(-r_preds)
        fake_loss = F.softplus(f_preds)

        return real_loss.mean() + fake_loss.mean()

    def dis_r1_loss(self, real_samps, netD, r1=10, d_reg_every=16):
        """
        :param real_samps:
        :param r1:
        :param d_reg_every:
        :return:
        """
        real_samps.requires_grad = True
        r_preds = netD(real_samps)

        grad_real, = th.autograd.grad(outputs=r_preds.sum(), inputs=real_samps, create_graph=True)

        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

        return r1 / 2 * grad_penalty * d_reg_every

    def gen_loss(self, real_samps, fake_samps, netD):
        """
        :param real_samps:
        :param fake_samps:
        :return:
        """
        f_preds = netD(fake_samps)

        return F.softplus(-f_preds).mean()
