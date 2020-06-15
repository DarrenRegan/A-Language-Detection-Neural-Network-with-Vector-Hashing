package ie.gmit.sw;
//Darren Regan - G00326934 - Group C - 4th yr AI Project
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Locale;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import ie.gmit.sw.*;

/*This class processes wili-2018-Small-11750-Edited.txt, takes each line in the txt file and writes it out in a set of numbers
* These numbers are then going to be read in a csv file
* Steps: Look over "Text",                           
* For each n-gram, compute the index = ngram.hashCode() % vector.length
* vector[index]++ Increment vector at index by 1
* NORAMLIZE THE VECTORS VALUES
* Utilities.normalize(vector, -1, 1);
* WRITE OUT VECTOR TO A CSV FILE using df.format(vector[index])
* WRITE OUT THE LANGUAGE NUMBERS TO THE SAME ROW IN THE CSV FILE
* LAST STEP CLASSIFACTION
* There is 234 Languages in the txt file, using language.values() shows each number.
* If for example 5 was the correct answer Arabic would be the answer, which would be written out like (0, 0, 0, 0, 1, 0, ...0)etc
* Vector.length + labels (Number of Elements in each row) Size of Vector + 235 Labels
* REFERENCES EXTREMELY USEFUL: Alot of code is mix and match between these links and labs on moodle
* 1. https://www.heatonresearch.com/encog/
* 2. https://s3.amazonaws.com/heatonresearch-books/free/encog-3_3-quickstart.pdf
* 3. https://s3.amazonaws.com/heatonresearch-books/free/Encog3Java-User.pdf
* 4. https://github.com/jeffheaton/encog-java-examples/tree/master/src/main/java/org/encog/examples
* 5. https://s3.amazonaws.com/heatonresearch-books/free/encog-3_3-devguide.pdf
*/
public class VectorProcessor {
	//Variables
	private final int NUMBER_OF_LANGUAGES = Language.values().length; //235 languages
	private double[] index = new double[NUMBER_OF_LANGUAGES];
	private double[] vector;
	private String[] record; // Used to remove dodgy lines of text...
	private String line, text, language;
	private int i, j, ngramSize = 0;
	private DecimalFormat dft = new DecimalFormat("###.###"); // Three places of percision
	private DecimalFormat dfl = new DecimalFormat("#.#"); 
	NumberFormat nf = NumberFormat.getNumberInstance(Locale.ENGLISH);
	private Language[] langs = Language.values();
	private File file = new File("wili-2018-Small-11750-Edited.txt");
	private File csvOutputFile  = new File("data.csv");
	private BufferedReader br;
	private BufferedWriter bw;
	private FileWriter fw;    
	
	//Constructors
	public VectorProcessor() {
		//Default Constructor
	}
	
	public VectorProcessor(int ngramSize, int vectorSize) { // TODO: make ngramSize & vectorSize be defined by user in menu instead of set here by default
		// If data.csv exists then delete it
		if (csvOutputFile.exists()) {
			csvOutputFile.delete();
		}
		this.ngramSize = ngramSize;
		vector = new double[vectorSize];
	}
	
	/* Parse File and call process() on EACH line, process will then go through the line and return if its a language that exists in the enum class Language
	 * If for example 5 was the correct answer Arabic would be the answer, which would be written out like (0, 0, 0, 0, 1, 0, ...0)etc
	 * Vector.length + labels (Number of Elements in each row) Size of Vector + 235 Labels
	 * */
	public void parse() throws Exception {
		System.out.print("\n Reading Dataset... \n");

		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
			//String line;
			System.out.println("\n Your Dataset file has been read and parsed!");
			System.out.println("\n Please wait while we process... \n ");
			while ((line = br.readLine()) != null) {
				process(line);
			}
		} catch (IOException e) {
			e.getMessage();
		}
	}//parse()
	
	
	public void process(String line) throws Exception {
		record = line.split("@");
		// Used to remove dodgy lines of text...
		if (record.length > 2) {
			return;
		}
		text = record[0].toUpperCase();
		language = record[1];
		
		if (!Language.isEnum(language, Language.class)) {
			return;
		}
		
		// Sets vector index i to 0 on each iteration. 
		for (i = 0; i < vector.length; i++) {
			vector[i] = 0;
		}
		
		// List of n-grams
		List<String> ngrams = ngram(text, ngramSize);
		
		//For each n-gram, compute the index = ngram.hashCode() % vector.length and increment the vector index by 1
		for (String ngram : ngrams) {
			vector[ngram.hashCode() % vector.length]++;
		}

		// Normalize Vector Values
		vector = Utilities.normalize(vector, 0.0, 1.0);

		index = languageToVector(Language.valueOf(language));

		// Save normalized vector values to CSV file
		try {
			fw = new FileWriter(csvOutputFile, true);

			for (i = 0; i < vector.length; i++) {
				fw.append(dft.format(vector[i]) + ",");
			}
			
			for (j = 0; j < index.length; j++) {
                fw.append(dfl.format(index[j]) + ",");
            }
		} catch (Exception exception) {
			exception.printStackTrace();
		} finally {
			fw.append("\n");
			fw.flush();
			fw.close();
		}
	}//process()
	
	
	// Creates an n-gram from the line of text
	public static List<String> ngram(String text, int ngramSize) {
		List<String> ngrams = new ArrayList<String>();
		int i;
		
        for (i = 0; i < text.length() - ngramSize + 1; i++) {
            ngrams.add(text.substring(i, i + ngramSize));
        }
        
        return ngrams;
	}//ngram()
	
	public double[] languageToVector(Language language) throws IOException {
		double[] languages = new double[Language.values().length];

		Language[] langs = Language.values();

		for (i = 0; i < langs.length; i++) {
			if (language == langs[i]) {
				languages[i] = 1.0;
			}
		}

		return languages;
	}//languageToVector()
/*	public static void main(String[] args) throws Exception {
        VectorProcessor vectorProcessor = new VectorProcessor(510, 3);
        vectorProcessor.parse();
    }//main()*/
}
