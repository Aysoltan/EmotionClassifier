package corpus;

/**
 * This class sets the each token of tweet as a string. It has the object method
 * normalizeToken() which trims the white spaces at the begin and the end of the
 * token and converts it into upper case.
 * 
 * @author AysoltanGravina
 *
 */
public class Token {

	// Attributes
	private String token;

	// Getter and Setter
	public String getToken() {
		return token;
	}

	public void setToken(String token) {
		this.token = token;
	}

}
