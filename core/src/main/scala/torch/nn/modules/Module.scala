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

import org.bytedeco.javacpp.CharPointer
import org.bytedeco.pytorch
import org.bytedeco.pytorch.{Conv2dImpl, InputArchive, OutputArchive}
import torch.{DType, Device, Tensor}

import java.nio.CharBuffer
import scala.collection.immutable.{ArraySeq, SeqMap, TreeSeqMap}
import scala.reflect.ClassTag

abstract class Module {

  protected[torch] var _nativeModule = pytorch.Module()
  private[torch] def nativeModule: pytorch.Module = _nativeModule // = pytorch.Module()
  private var childModules: TreeSeqMap[String, Module] = TreeSeqMap.empty

  def namedBuffers(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val buffers = nativeModule.named_buffers(recurse)
    TreeSeqMap.from((0 until buffers.size().toInt).map { i =>
      val item = buffers.get(i)
      (item.key().getString(), Tensor.apply[DType](item.access()))
    })

  def namedParameters(recurse: Boolean = true): SeqMap[String, Tensor[?]] =
    val params = nativeModule.named_parameters(recurse)
    TreeSeqMap.from((0 until params.size().toInt).map { i =>
      val item = params.get(i)
      (item.key().getString(), Tensor.apply[DType](item.access()))
    })

  def parameters: Seq[Tensor[?]] = parameters(recurse = true)

  def parameters(recurse: Boolean): Seq[Tensor[?]] =
    ArraySeq.unsafeWrapArray(nativeModule.parameters().get).map(Tensor.apply[DType])

  // TODO make strict a parameter
  // TODO improve error handling
  def loadStateDict(stateDict: Map[String, Tensor[DType]]): Unit =
    val tensorsToLoad = namedParameters() ++ namedBuffers()
    // assert(stateDict.keySet -- tensorsToLoad.keySet == Set.empty, s"keys missing in state dict: ${tensorsToLoad.keySet -- stateDict.keySet}")
    for ((key, param) <- tensorsToLoad if stateDict.contains(key))
      noGrad {
        param.copy_(stateDict(key))
      }

  def modules(recurse: Boolean): Seq[Module] =
    childModules.values.flatMap(child => child +: child.modules).toSeq.distinct
  def modules: Seq[Module] = modules(recurse = true)

  def namedChildren: SeqMap[String, Module] = childModules
  def namedModules: SeqMap[String, Module] =
    namedChildren.flatMap((name, module) => module.namedModules)

  def register[M <: Module](child: M)(using name: sourcecode.Name) =
    // println(s"registering ${name.value}:$child")
    childModules = childModules.updated(name.value, child)
    nativeModule.register_module(name.value, child.nativeModule)
    child

  def register[D <: DType](t: Tensor[D], requiresGrad: Boolean = true)(using
      name: sourcecode.Name
  ): Tensor[D] =
    nativeModule.register_parameter(name.value, t.native, requiresGrad)
    t

  def eval(): Unit = nativeModule.eval()

  def isTraining: Boolean = nativeModule.is_training

  def train(on: Boolean = true): Unit = nativeModule.train(on)

  def to(device: Device): this.type =
    nativeModule.to(device.toNative, false)
    this

  def save(outputArchive: OutputArchive) = nativeModule.save(outputArchive)

  def load(inputArchive: InputArchive) = nativeModule.load(inputArchive)

  override def toString(): String = getClass().getSimpleName()

  private def doSummarize(indent: Int): String =
    val thisModule = toString
    if modules.isEmpty then thisModule
    else
      thisModule + namedChildren
        .map((name, module) => s"${" " * (indent + 2)}($name): " + module.doSummarize(indent + 2))
        .mkString("(\n", "\n", s"\n${" " * indent})")
  def summarize: String =
    doSummarize(0)
}

trait HasParams[ParamType <: FloatNN | ComplexNN: Default] extends Module:
  override def parameters(recurse: Boolean): Seq[Tensor[ParamType]] =
    nativeModule.parameters(recurse).get().toSeq.map(Tensor.apply[ParamType])
  override def parameters: Seq[Tensor[ParamType]] = parameters(recurse = true)
  transparent inline def paramType: DType = summon[Default[ParamType]].dtype

trait HasWeight[ParamType <: FloatNN | ComplexNN]:
  def weight: Tensor[ParamType]

/** Transforms a single tensor into another one of the same type. */
trait TensorModule[D <: DType] extends Module with (Tensor[D] => Tensor[D]):
  override def toString(): String = "TensorModule"

trait TensorModuleBase[D <: DType, D2 <: DType] extends Module with (Tensor[D] => Tensor[D2]) {
  override def toString() = "TensorModuleBase"
}
