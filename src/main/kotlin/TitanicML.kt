import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.*
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Column
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataTypes
import org.jetbrains.spark.api.sparkContext
import org.jetbrains.spark.api.withSpark

private const val DATA_DIRECTORY = "/sparkdata"

fun main() {
    withSpark(
        master = "local[2]",
        appName = "Titanic_ML"
    ) {
        spark.sparkContext.setLogLevel("ERROR")

        val xlsPassengers: Dataset<Row> = spark.read()
            .option("delimiter", ";")
            .option("inferSchema", "true")
            .option("header", "true")
            .csv("$DATA_DIRECTORY/titanic.csv")

        val castedPassengers: Dataset<Row> = xlsPassengers
            .withColumn("survived", Column("survived").cast(DataTypes.DoubleType))
            .withColumn("pclass", Column("pclass").cast(DataTypes.DoubleType))
            .withColumn("sibsp", Column("sibsp").cast(DataTypes.DoubleType))
            .withColumn("parch", Column("parch").cast(DataTypes.DoubleType))
            .withColumn("age", Column("age").cast(DataTypes.DoubleType))
            .withColumn("fare", Column("fare").cast(DataTypes.DoubleType))

        castedPassengers.printSchema()

        castedPassengers.show()

        val passengers: Dataset<Row> = castedPassengers
            .select("survived", "pclass", "sibsp", "parch", "sex", "embarked", "age", "fare", "name")

        val split: Array<Dataset<Row>> = passengers.randomSplit(doubleArrayOf(0.7, 0.3), 12345)

        val training: Dataset<Row> = split[0].cache()
        val test: Dataset<Row> = split[1].cache()

        val regexTokenizer = RegexTokenizer()
            .setInputCol("name")
            .setOutputCol("name_parts")
            .setPattern("\\w+").setGaps(false)

        val remover = StopWordsRemover()
            .setStopWords(arrayOf("mr", "mrs", "miss", "master", "jr", "j", "c", "d"))
            .setInputCol("name_parts")
            .setOutputCol("filtered_name_parts")

        val hashingTF: HashingTF = HashingTF()
            .setInputCol("filtered_name_parts")
            .setOutputCol("text_features")
            .setNumFeatures(1000)

        val sexIndexer = StringIndexer()
            .setInputCol("sex")
            .setOutputCol("sexIndexed")
            .setHandleInvalid("keep") // special mode to create special double value for null values

        val embarkedIndexer = StringIndexer()
            .setInputCol("embarked")
            .setOutputCol("embarkedIndexed")
            .setHandleInvalid("keep") // special mode to create special double value for null values

        val imputer = Imputer()
            .setInputCols(arrayOf("pclass", "sibsp", "parch", "age", "fare"))
            .setOutputCols(
                arrayOf(
                    "pclass_imputed",
                    "sibsp_imputed",
                    "parch_imputed",
                    "age_imputed",
                    "fare_imputed"
                )
            )
            .setStrategy("mean")

        val assembler = VectorAssembler()
            .setInputCols(
                arrayOf(
                    "pclass_imputed",
                    "sibsp_imputed",
                    "parch_imputed",
                    "age_imputed",
                    "fare_imputed",
                    "sexIndexed",
                    "embarkedIndexed"
                )
            )
            .setOutputCol("features")

        val polyExpansion = PolynomialExpansion()
            .setInputCol("features")
            .setOutputCol("polyFeatures")
            .setDegree(2)

        // We should join together text features and number features into one vector
        val assembler2 = VectorAssembler()
            .setInputCols(arrayOf("polyFeatures", "text_features"))
            .setOutputCol("joinedFeatures")

        val scaler = MinMaxScaler() // new MaxAbsScaler()
            .setInputCol("joinedFeatures")
            .setOutputCol("unnorm_features")

        val normalizer: Normalizer = Normalizer()
            .setInputCol("unnorm_features")
            .setOutputCol("norm_features")
            .setP(1.0)

        val pca: PCA = PCA()
            .setInputCol("norm_features")
            .setK(100)
            .setOutputCol("pca_features")

        val trainer = RandomForestClassifier()
            .setLabelCol("survived")
            .setFeaturesCol("pca_features")
            .setMaxDepth(20)
            .setNumTrees(200)

        val pipeline = Pipeline()
            .setStages(
                arrayOf(
                    regexTokenizer,
                    remover,
                    hashingTF,
                    sexIndexer,
                    embarkedIndexer,
                    imputer,
                    assembler,
                    polyExpansion,
                    assembler2,
                    scaler,
                    normalizer,
                    pca,
                    trainer
                )
            )

        val evaluator = MulticlassClassificationEvaluator()
            .setLabelCol("survived")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")

        val paramGrid = ParamGridBuilder()
            .addGrid(hashingTF.numFeatures(), intArrayOf(100, 1000))
            .addGrid(pca.k(), intArrayOf(10, 100))
            .build()

        val cv = CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(evaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(3)

        // Run cross-validation, and choose the best set of parameters.
        val model = cv.fit(training)
        /*println("---------- The best model's parameters are ----------")
        println(
            "Num of features " + ((model.bestModel() as PipelineModel).stages()[2] as HashingTF).getNumFeatures()
        )
        println(
            "Amount of components in PCA " + ((model.bestModel() as PipelineModel).stages()[11] as PCAModel).getK()
        )*/
        val rawPredictions: Dataset<Row> = model.transform(test)
        val accuracy = evaluator.evaluate(rawPredictions)
        println("Test Error = " + (1.0 - accuracy))
    }
}