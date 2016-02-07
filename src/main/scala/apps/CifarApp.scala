package apps

import java.io._

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.bytedeco.javacpp.caffe._

import libs._
import loaders._
import preprocessing._

// for this app to work, $SPARKNET_HOME should be the SparkNet root directory
// and you need to run $SPARKNET_HOME/caffe/data/cifar10/get_cifar10.sh
object CifarApp {
  val trainBatchSize = 100
  val testBatchSize = 100
  val channels = 3
  val width = 32
  val height = 32
  val imShape = Array(channels, height, width)
  val size = imShape.product

  val workerStore = new WorkerStore()

  def main(args: Array[String]) {
    val numWorkers = args(0).toInt
    val conf = new SparkConf()
      .setAppName("Cifar")
      .set("spark.driver.maxResultSize", "5G")
      .set("spark.task.maxFailures", "1")
      .setExecutorEnv("LD_LIBRARY_PATH", "/usr/local/cuda-7.5/lib64:/root/javacpp-presets/caffe/cppbuild/linux-x86_64/caffe-master/build/lib")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    val sparkNetHome = sys.env("SPARKNET_HOME")

    // information for logging
    val startTime = System.currentTimeMillis()
    val trainingLog = new PrintWriter(new File(sparkNetHome + "/training_log_" + startTime.toString + ".txt" ))
    def log(message: String, i: Int = -1) {
      val elapsedTime = 1F * (System.currentTimeMillis() - startTime) / 1000
      if (i == -1) {
        trainingLog.write(elapsedTime.toString + ": "  + message + "\n")
      } else {
        trainingLog.write(elapsedTime.toString + ", i = " + i.toString + ": "+ message + "\n")
      }
      trainingLog.flush()
    }

    val loader = new CifarLoader(sparkNetHome + "/caffe/data/cifar10/")
    log("loading train data")
    var trainRDD = sc.parallelize(loader.trainImages.zip(loader.trainLabels))
    log("loading test data")
    var testRDD = sc.parallelize(loader.testImages.zip(loader.testLabels))

    // convert to dataframes
    val schema = StructType(StructField("data", ArrayType(FloatType), false) :: StructField("label", IntegerType, false) :: Nil)
    var trainDF = sqlContext.createDataFrame(trainRDD.map{ case (a, b) => Row(a.map(x => x.toFloat), b)}, schema)
    var testDF = sqlContext.createDataFrame(testRDD.map{ case (a, b) => Row(a.map(x => x.toFloat), b)}, schema)

    log("repartition data")
    trainDF = trainDF.repartition(numWorkers).cache()
    testDF = testDF.repartition(numWorkers).cache()

    val numTrainData = trainDF.count()
    log("numTrainData = " + numTrainData.toString)

    val numTestData = testDF.count()
    log("numTestData = " + numTestData.toString)

    val trainPartitionSizes = trainDF.mapPartitions(iter => Array(iter.size).iterator).persist()
    val testPartitionSizes = testDF.mapPartitions(iter => Array(iter.size).iterator).persist()
    trainPartitionSizes.foreach(size => workerStore.put("trainPartitionSize", size))
    testPartitionSizes.foreach(size => workerStore.put("testPartitionSize", size))
    log("trainPartitionSizes = " + trainPartitionSizes.collect().deep.toString)
    log("testPartitionSizes = " + testPartitionSizes.collect().deep.toString)

    val workers = sc.parallelize(Array.range(0, numWorkers), numWorkers)

    // initialize nets on workers
    workers.foreach(_ => {
      val netParam = new NetParameter()
      ReadProtoFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick.prototxt", netParam)

      val solverParam = new SolverParameter()
      ReadSolverParamsFromTextFileOrDie(sparkNetHome + "/models/cifar10/cifar10_quick_solver.prototxt", solverParam)
      solverParam.clear_net()
      solverParam.set_allocated_net_param(netParam)

      val solver = new CaffeSolver(solverParam, schema, new DefaultPreprocessor(schema))
      workerStore.put("netParam", netParam) // prevent netParam from being garbage collected
      workerStore.put("solverParam", solverParam) // prevent solverParam from being garbage collected
      workerStore.put("solver", solver)
    })

    // initialize weights on master
    var netWeights = workers.map(_ => workerStore.get[CaffeSolver]("solver").trainNet.getWeights()).collect()(0)

    var i = 0
    while (true) {
      log("broadcasting weights", i)
      val broadcastWeights = sc.broadcast(netWeights)
      log("setting weights on workers", i)
      workers.foreach(_ => workerStore.get[CaffeSolver]("solver").trainNet.setWeights(broadcastWeights.value))

      if (i % 5 == 0) {
        log("testing", i)
        val testAccuracies = testDF.mapPartitions(
          testIt => {
            val numTestBatches = workerStore.get[Int]("testPartitionSize") / testBatchSize
            var accuracy = 0F
            for (j <- 0 to numTestBatches - 1) {
              val out = workerStore.get[CaffeSolver]("solver").trainNet.forward(testIt)
              accuracy += out("accuracy").get(Array())
            }
            Array[Float](accuracy / numTestBatches).iterator
          }
        ).cache()
        val accuracy = testAccuracies.sum / numWorkers
        log("%.2f".format(100F * accuracy) + "% accuracy", i)
      }

      log("training", i)
      val syncInterval = 10
      trainDF.foreachPartition(
        trainIt => {
          val t1 = System.currentTimeMillis()
          val r = scala.util.Random
          val len = workerStore.get[Int]("trainPartitionSize")
          val startIdx = r.nextInt(len - syncInterval * trainBatchSize)
          val it = trainIt.drop(startIdx)
          val t2 = System.currentTimeMillis()
          print("stuff took " + ((t2 - t1) * 1F / 1000F).toString + " s\n")
          for (j <- 0 to syncInterval - 1) {
            workerStore.get[CaffeSolver]("solver").step(it)
          }
          val t3 = System.currentTimeMillis()
          print("iters took " + ((t3 - t2) * 1F / 1000F).toString + " s\n")
        }
      )

      log("collecting weights", i)
      netWeights = workers.map(_ => { workerStore.get[CaffeSolver]("solver").trainNet.getWeights() }).reduce((a, b) => WeightCollection.add(a, b))
      netWeights.scalarDivide(1F * numWorkers)
      log("weight = " + netWeights.allWeights("conv1")(0).toFlat()(0).toString, i)
      i += 1
    }

    log("finished training")
  }
}
