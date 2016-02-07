package apps

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._
import preprocessing._

object TempApp {
  def main(args: Array[String]) {

    val sparkNetHome = "/root/SparkNet/"

    val batch = new Array[Row](100)
    for (i <- 0 to 100 - 1) {
      val im = new Array[Float](32 * 32 * 3).toList
      val label = 0
      batch(i) = Row(im, label)
    }

    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)

    val netParam = new NetParameter()
    ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick.prototxt", netParam)

    val solverParam = new SolverParameter()
    ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_solver.prototxt", solverParam)
    solverParam.clear_net()
    solverParam.set_allocated_net_param(netParam)

    val solver = new CaffeSolver(solverParam, schema, new DefaultPreprocessor(schema))

    val t1 = System.currentTimeMillis()
    for (i <- 0 to 10 - 1) {
      solver.step(batch.iterator)
    }
    val t2 = System.currentTimeMillis()
    print("iters took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
  }
}
