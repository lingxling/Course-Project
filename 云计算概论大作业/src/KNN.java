import java.util.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KNN{
    //这个类主要用来保存训练样例和测试样例的距离以及训练样例自身的标签。
    public static class DistanceLabel implements Comparable<DistanceLabel>{
        public int distance;
        public String label;
        public DistanceLabel(){
            distance = Integer.MAX_VALUE;
            label = null;
        }
        public DistanceLabel(String str, int dis){
            distance = dis;
            label = str;
        }
        public DistanceLabel(int dis, String str){
            distance = dis;
            label = str;
        }
        public int compareTo(DistanceLabel other){
            return this.distance - other.distance;
        }
    }
    public static class KnnMapper extends Mapper<Object, Text, IntWritable, Text>{
        private static int testId = 0;
        private List<int[]> testSamples = new ArrayList<int[]>();
        private List<List<DistanceLabel>> distanceLabels = new ArrayList<List<DistanceLabel>>();
        private final int K = 100;
        private final int featuresNum = 784;  //28*28
        
        private int getDistance(int[] s1_arr, int[] s2_arr, int n) {
            int distance = 0;
            for (int i = 0; i < n; i++) {
                distance = distance + (s1_arr[i] - s2_arr[i])*(s1_arr[i] - s2_arr[i]);  //求欧氏距离
            }
            return distance;
        }
        @Override
        public void setup(Context context) throws IOException, InterruptedException {            
            //读入测试文件，保存在testSamples中
            Configuration conf = context.getConfiguration();
            Path pt = new Path(conf.get("testFile"));
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));
            String line;
            while ((line = br.readLine()) != null) {
		        int[] curTestSample = new int [featuresNum];
                StringTokenizer st = new StringTokenizer(line, ",");
                for (int i = 0; i < featuresNum; ++i){
                    curTestSample[i] = Integer.parseInt(st.nextToken());
                }
                testSamples.add(curTestSample);
                testId ++;
            } 
            br.close();

            //初始化distanceLabel
            //distanceLabels用于保存每个测试样例和所有训练样例的距离、训练样例自身标签
            for (int i = 0; i < testId; ++i){
                distanceLabels.add(new ArrayList<DistanceLabel>());
            }
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
            String curTrainSample = value.toString();
            StringTokenizer st = new StringTokenizer(curTrainSample, ",");
            String curLabel = st.nextToken();
            int[] curTrainArr = new int [featuresNum];
            for (int i = 0; i < featuresNum; ++i){
                curTrainArr[i] = Integer.parseInt(st.nextToken());
            }
            for (int i = 0; i < testId; ++i){
                //计算该训练样例与每个测试样例的距离并保存
                int curDistance = getDistance(testSamples.get(i), curTrainArr, featuresNum);
                distanceLabels.get(i).add(new DistanceLabel(curDistance, curLabel));
            }
        }
        @Override
        protected void cleanup (Context context) throws IOException, InterruptedException{
            for (int i = 0; i < testId; ++i){
                Collections.sort(distanceLabels.get(i));  //排序然后选出排在前K的label
                for (int j = 0; j < K; ++j){
                    context.write(new IntWritable(i), new Text(distanceLabels.get(i).get(j).label));
                }
            }
        }
    }

    //input key: 测试样例id, input value: 与测试样例距离近样例的标签
    //output key: 测试样例id, output value: 测试样例预测标签
    public static class KnnReducer extends Reducer<IntWritable, Text, IntWritable, Text>{
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            HashMap<String, Integer> labelCounter = new HashMap<String, Integer>();
            for (Text val : values){
                if (labelCounter.containsKey(val.toString())){
                    labelCounter.put(val.toString(), labelCounter.get(val.toString())+1);
                }
                else{
                    labelCounter.put(val.toString(), 1);
                }
            }

            //选出距离在前K的标签中频数最高的标签
            int maxFreq = -1;
            String mostFreqLabel = null;
            for (Map.Entry<String, Integer> entry: labelCounter.entrySet()){
                if (entry.getValue() > maxFreq){
                    mostFreqLabel = entry.getKey();
                    maxFreq = entry.getValue();
                }
            }
            context.write(key, new Text(mostFreqLabel));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration config = new Configuration();
        config.set("testFile", args[2]);  //测试文件
        Job job = new Job(config, "K Nearest Neighbor");
   
        job.setJarByClass(KNN.class);
        job.setMapperClass(KnnMapper.class);
        job.setReducerClass(KnnReducer.class);
        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));  //训练文件
        FileOutputFormat.setOutputPath(job, new Path(args[1]));  //预测输出文件
        System.exit(job.waitForCompletion(true) ? 0 : 1);        
    }
}
