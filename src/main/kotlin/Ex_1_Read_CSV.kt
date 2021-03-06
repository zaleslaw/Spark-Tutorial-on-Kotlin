import org.apache.hadoop.log.LogLevel
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataTypes
import org.jetbrains.spark.api.*

private const val DATA_DIRECTORY = "/sparkdata"

data class StateNamesRow(
    val id: Int,
    val name: String,
    val year: Int,
    val gender: String,
    val state: String,
    val count: Int
)

data class NationalNamesRow(val Id: Int, val Name: String, val Year: Int, val Gender: String, val Count: Int)

fun main() {
    withSpark(
        master = "local[2]",
        appName = "CSV_to_Parquet_JSON"
    ) {

        spark.sparkContext.setLogLevel("ERROR")

        // Step - 1: Extract the schema
        // Read CSV and automatically extract the schema
        val stateNames: Dataset<Row> = spark.read()
            .option("header", "true")
            .option("encoding", "windows-1251")
            .option("inferSchema", "true") // Id as int, count as int due to one extra pass over the data
            .csv("$DATA_DIRECTORY/StateNames.csv")

        stateNames.show()
        stateNames.printSchema()

        // Step - 2: In reality it can be too expensive and CPU-burst
        // If dataset is quite big, you can infer schema manually
        val fields = arrayOf(
            DataTypes.createStructField("Id", DataTypes.IntegerType, true),
            DataTypes.createStructField("Name", DataTypes.StringType, true),
            DataTypes.createStructField("Year", DataTypes.IntegerType, true),
            DataTypes.createStructField("Gender", DataTypes.StringType, true),
            DataTypes.createStructField("Count", DataTypes.IntegerType, true)
        )

        val nationalNamesSchema = DataTypes.createStructType(fields)

        val stateNamesDS = stateNames.downcast<Row, StateNamesRow>()

        stateNamesDS.filter {
            it.year == 1900
        }.map {
            it.year
        }.show()


        val nationalNames: Dataset<Row> = spark.read()
            .option("header", "true")
            .schema(nationalNamesSchema)
            .csv("$DATA_DIRECTORY/NationalNames.csv")

        nationalNames.show()
        nationalNames.printSchema()

        nationalNames.withCached {
            // Step - 3: Simple dataframe operations
            // Filter & select & orderBy
            nationalNames
                .where("Gender == 'M'")
                .select("Name", "Year", "Count")
                .orderBy("Name", "Year")
                .show(100)

            // Registered births by year in US since 1880
            nationalNames
                .groupBy("Year")
                .sum("Count").`as`("Sum")
                .orderBy("Year")
                .show(200)
        }
    }
}
