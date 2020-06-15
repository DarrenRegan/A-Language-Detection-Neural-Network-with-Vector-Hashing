package ie.gmit.sw;

public class Kmer {
	
	private int kmer;
    private String kmerString;

    public Kmer(String kmerString) {
        this.kmerString = kmerString;
        this.kmer = kmerString.hashCode();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Kmer)) return false;
        Kmer kmer1 = (Kmer) o;
        return kmer == kmer1.kmer;
    }

    @Override
    public int hashCode() {
        return kmer;
    }

    @Override
    public String toString() {
        return "Kmer{" +
                "kmerString='" + kmerString + '\'' +
                ", kmer=" + kmer +
                '}';
	    }
}
