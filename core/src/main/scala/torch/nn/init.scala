/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package torch
package nn

import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.FanModeType
import org.bytedeco.pytorch.kFanIn
import org.bytedeco.pytorch.kFanOut
import org.bytedeco.pytorch.Nonlinearity as NonlinearityNative
import org.bytedeco.pytorch.kLinear
import org.bytedeco.pytorch.kConv1D
import org.bytedeco.pytorch.kConv2D
import org.bytedeco.pytorch.kConv3D
import org.bytedeco.pytorch.kConvTranspose1D
import org.bytedeco.pytorch.kConvTranspose2D
import org.bytedeco.pytorch.kConvTranspose3D
import org.bytedeco.pytorch.kSigmoid
import org.bytedeco.pytorch.kReLU
import org.bytedeco.pytorch.kLeakyReLU
import org.bytedeco.pytorch.Scalar

// TODO implement remaining init functions
object init:

  /**
    * Fills the input Tensor with values drawn from the uniform distribution U(a,b)U(a,b).
    *
    * @param t – an n-dimensional torch.Tensor
    * @param a – the lower bound of the uniform distribution
    * @param b – the upper bound of the uniform distribution
    */  
  def uniform_(
      t: Tensor[?],
      a: Double = 0,
      b: Double = 0
  ): Unit =
    torchNative.uniform_(t.native, a, b)


  // TODO: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a105c2a8ef81c6faa82a01cf35ce9f3b1.html
  // TODO: https://pytorch.org/cppdocs/api/file_torch_csrc_api_include_torch_nn_init.h.html#file-torch-csrc-api-include-torch-nn-init-h
  /**
    * 
    * Fills the input Tensor with values drawn from the normal distribution N(mean,std2)N(mean,std2).
    *
    * @param t – an n-dimensional torch.Tensor
    * @param mean – the mean of the normal distribution
    * @param std – the standard deviation of the normal distribution
    */  
  def normal_(
      t: Tensor[?],
      mean: Double = 0,
      std: Double = 0
  ): Unit =
    torchNative.normal_(t.native, mean, std)

  // TODO valid for all scala types
  /**
    * Fills the input Tensor with the value valval.
    *
    * @param t – an n-dimensional torch.Tensor
    * @param fillValue – the value to fill the tensor with
    */
  def constant_(t: Tensor[?], fillValue: Double): Unit =
    torchNative.constant_(t.native, Scalar(fillValue)): Unit

  /**
    * Fills the input Tensor with the scalar value 1.
    *
    * @param t – an n-dimensional torch.Tensor
    */
  def ones_(
      t: Tensor[?]
  ): Unit =
    torchNative.ones_(t.native)


  /**
    * Fills the input Tensor with the scalar value 0.
    *
    * @param t – an n-dimensional torch.Tensor
    */
  def zeros_(
      t: Tensor[?]
  ): Unit =
    torchNative.zeros_(t.native)

  /**
    * Fills the 2-dimensional input [[Tensor]] with the identity matrix. Preserves the identity of 
    * the inputs in Linear layers, where as many inputs are preserved as possible.
    *
    * @param t – a 2-dimensional torch.Tensor
    */
  def eye_(
      t: Tensor[?]
  ): Unit =
    torchNative.eye_(t.native)
  
  
  // TODO: no groups available
  /**
    * ills the {3, 4, 5}-dimensional input [[Tensor]] with the Dirac delta function. Preserves the identity 
    * of the inputs in Convolutional layers, where as many input channels are preserved as possible. In case 
    * of groups>1, each group of channels preserves identity
    * 
    * @param t – a {3, 4, 5}-dimensional torch.Tensor
    * @param groups (int, optional) – number of groups in the conv layer (default: 1)
    */
  def dirac_(
      t: Tensor[?]
  ): Unit =
    torchNative.dirac_(t.native)
  

  /**
    * Fills the input [[Tensor]] with values according to the method described in Understanding the difficulty
    * of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform 
    * distribution. The resulting tensor will have values sampled from $U(−a,a)$ where
    *  $a=gain \times \sqrt{\frac{6}{fan_in+fan_out$}}
    * 
    * Also known as Glorot initialization.
    *
    * @param t – an n-dimensional torch.Tensor
    * @param gain – an optional scaling factor
    */
  def xavier_normal_(
      t: Tensor[?],
      gain: Double = 1.0
  ): Unit =
    torchNative.xavier_normal_(t.native, gain)
  

  /**
    * Fills the input Tensor with values according to the method described in Delving deep into 
    * rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), 
    * using a uniform distribution. The resulting tensor will have values sampled from $U(−bound,bound)$
    * where
    * $\text{bound} = \text{gain} \times \sqrt{\frac{3}{fan_mode}}
    * 
    * Also known as He initialization.
    * 
    * @param t – an n-dimensional torch.Tensor
    * @param a – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
    * @param mode – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
    * @param nonlinearity – the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
    */
  def kaimingUniform_(
      t: Tensor[?],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): Unit =
    torchNative.kaiming_uniform_(t.native, a, mode.toNative, nonlinearity.toNative)


/**
  * Fills the input Tensor with values according to the method described in "Delving deep into rectifiers: 
  * Surpassing human-level performance on ImageNet classification - He, K. et al. (2015), using a normal 
  * distribution. The resulting tensor will have values sampled from $N(0,std^22)$ where: 
  *    $$std = \frac{gain}{\sqrt{fan_mode}}
  *
  * Also known as He initialization.
  * 
  * @param t – an n-dimensional torch.Tensor
  * @param a – the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
  * @param mode – either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
  * @param nonlinearity – the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
  */
  def kaimingNormal_(
      t: Tensor[?],
      a: Double = 0,
      mode: Mode = Mode.FanIn,
      nonlinearity: NonLinearity = NonLinearity.LeakyReLU
  ): Unit =
    torchNative.kaiming_normal_(t.native, a, mode.toNative, nonlinearity.toNative)

  // TODO: no trunc normal. No docs found on C++
  // /**
  //   * Fills the input Tensor with values drawn from a truncated normal distribution. The values are 
  //   * effectively drawn from the normal distribution $N(\text{mean},\text{std}^2)$ with values outside 
  //   * $[a,b]$ redrawn until they are within the bounds. The method used for generating the random values
  //   * works best when $a \le \text{mean} \le \text{b}.
  //   *
  //   * @param t – an n-dimensional torch.Tensor
  //   * @param mean (float) – the mean of the normal distribution
  //   * @param std (float) – the standard deviation of the normal distribution
  //   * @param a (float) – the minimum cutoff value
  //   * @param b (float) – the maximum cutoff value
  //   */
  def trunc_(
      t: Tensor[?]
  ): Unit =
    torchNative.trunc_(t.native)

  // TODO: not implemented
  /**
    * Fills the input Tensor with a (semi) orthogonal matrix, as described in Exact solutions to the nonlinear 
    * dynamics of learning in deep linear neural networks - Saxe, A. et al. (2013). The input tensor must have
    * at least 2 dimensions, and for tensors with more than 2 dimensions the trailing dimensions are flattened.
    * 
    *
    * @param t – an n-dimensional torch.Tensor, where n≥2n≥2
    * @param gain – optional scaling factor
    */  
  def orthogonal_(
      t: Tensor[?],
      gain: Double = 1.0
  ): Unit =
    torchNative.orthogonal_(t.native, gain)

  // TODO: not implemented
  // TODO: see https://discuss.pytorch.org/t/torchvision-warnings-after-updating-pytorch-to-version-0-4-0/18453 (deprecated)
  /**
    * Fills the 2D input Tensor as a sparse matrix, where the non-zero elements will be drawn from the normal 
    * distribution $N(0,0.01)$, as described in Deep learning via Hessian-free optimization - Martens, J. (2010).
    *
    * @param t – an n-dimensional torch.Tensor
    * @param gain – The fraction of elements in each column to be set to zero
    * @param std – the standard deviation of the normal distribution used to generate the non-zero values
    */
  def sparse_(
      t: Tensor[?],
      sparsity: Double
  ): Unit =
    torchNative.sparse_(t.native, sparsity)


  enum Mode:
    case FanIn, FanOut
    private[torch] def toNative: FanModeType = FanModeType(this match
      case Mode.FanIn  => kFanIn()
      case Mode.FanOut => kFanOut()
    )

  enum NonLinearity:
    case Linear, Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D, Sigmoid,
      ReLU, LeakyReLU
    private[torch] def toNative: NonlinearityNative = NonlinearityNative(this match
      case NonLinearity.Linear          => kLinear()
      case NonLinearity.Conv1D          => kConv1D()
      case NonLinearity.Conv2D          => kConv2D()
      case NonLinearity.Conv3D          => kConv3D()
      case NonLinearity.ConvTranspose1D => kConvTranspose1D()
      case NonLinearity.ConvTranspose2D => kConvTranspose2D()
      case NonLinearity.ConvTranspose3D => kConvTranspose3D()
      case NonLinearity.Sigmoid         => kSigmoid()
      case NonLinearity.ReLU            => kReLU()
      case NonLinearity.LeakyReLU       => kLeakyReLU()
    )

