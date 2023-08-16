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
package modules
package embed

import org.bytedeco.pytorch
import org.bytedeco.pytorch.{EmbeddingImpl, EmbeddingOptions}

// format: off
/** A simple lookup table that stores embeddings of a fixed dictionary and size.
 * 
 * This module is often used to store word embeddings and retrieve them using 
 * indices. The input to the module is a list of indices, and the output is the 
 * corresponding word embeddings.
 * 
 * Shape:
 * - Input: $(∗)$, `IntTensor` or `LongTensor` of arbitrary shape containing 
 * the indices to extract
 * - Output: $(∗,H)$, where $*$ is the input shape and $H=\text{embedding_dimH}$
 *  
 * @note 1:
 * Keep in mind that only a limited number of optimizers support sparse 
 * gradients: currently it’s `optim.SGD` (CUDA and CPU), `optim.SparseAdam` 
 * (CUDA and CPU) and `optim.Adagrad` (CPU)
 * 
 * @note 2:
 * When `max_norm` is not `None`, Embedding’s forward method will modify the 
 * `weight` tensor in-place. Since tensors needed for gradient computations 
 * cannot be modified in-place, performing a differentiable operation on 
 * `Embedding.weight` before calling [[nn.Embedding]]’s forward method requires 
 * cloning `Embedding.weight` when `max_norm` is not `None`. For example:
 *    {{{
 *    import torch.nn
 * 
 *    val n, d, m = 3, 5, 7
 *    val embedding = nn.Embedding(n, d, max_norm=True)
 *    val W = torch.randn((m, d), requires_grad=True)
 *    val idx = torch.tensor(Seq(1, 2))
 *    val a = embedding.weight.clone() @ W.t()  // weight must be cloned for this to be differentiable
 *    val b = embedding(idx) @ W.t()  // modifies weight in-place
 *    val out = a.unsqueeze(0) + b.unsqueeze(1)
 *    val loss = out.sigmoid().prod()
 *    loss.backward()
 *    }}}
 * 
 * @example
 *
 * ```scala
 * import torch.nn
 * 
 * // an Embedding module containing 10 tensors of size 3
 * val embedding = nn.Embedding(10, 3)
 * // a batch of 2 samples of 4 indices each
 * val input = torch.LongTensor(Seq(Seq(1, 2, 4, 5)), Seq(4, 3, 2, 9)))
 * embedding(input)
 * ```
 * 
 * @param num_embeddings
 *  size of the dictionary of embeddings
 * @param embedding_dim
 *  the size of each embedding vector
 * @param padding_idx
 *  If specified, the entries at `padding_idx` do not contribute to the gradient;
 * therefore, the embedding vector at `padding_idx` is not updated during training, 
 * i.e. it remains as a fixed “pad”. For a newly constructed Embedding, the 
 * embedding vector at `padding_idx` will default to all zeros, but can be 
 * updated to another value to be used as the padding vector.
 * @param max_norm
 *  If given, each embedding vector with norm larger than max_norm is 
 * renormalized to have norm `max_norm`
 * @param norm_type
 * The p of the p-norm to compute for the `max_norm` option. Default 2
 * @param scale_grad_by_freq
 *  If given, this will scale gradients by the inverse of frequency of the words
 * in the mini-batch. Default `False`
 * @param sparseIf `true`, gradient w.r.t. `weight` matrix will be a sparse tensor. 
 * See Notes for more details regarding sparse gradients
 * 
 * @param weight (Tensor) – the learnable weights of the module of shape 
 * (`num_embeddings`, `embedding_dim`) initialized from $\mathcal{N}(0,1)$
 *
 * @group nn_embed
 */
// format: on
final class Embedding[D <: DType: Default](
                            num_embeddings: Long, 
                            embedding_dim: Long,
                            padding_idx: Option[Long] = None,
                            max_norm: Option[Double] = None,
                            norm_type: Option[Double] = Some(2),
                            scale_grad_by_freq: Option[Boolean] = Some(false),
                            sparse: Option[Boolean] = Some(false),
                            )
    extends TensorModule[D]:

  private val options = EmbeddingOptions(num_embeddings, embedding_dim)
  padding_idx.foreach(options.padding_idx().put)
  max_norm.foreach(options.max_norm().put)
  norm_type.foreach(options.norm_type().put)
  scale_grad_by_freq.foreach(options.scale_grad_by_freq().put)
  sparse.foreach( options.sparse().put)
  // _weight();


  override val nativeModule: EmbeddingImpl = EmbeddingImpl(options)

  def apply(t: Tensor[D]): Tensor[D] = Tensor(nativeModule.forward(t.native))

  override def toString = getClass().getSimpleName()
