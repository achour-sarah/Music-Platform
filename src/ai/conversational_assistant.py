import os
import sqlite3
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (.env) explicitly by path
env_path = r"c:\SDV\Music\.env"
load_dotenv(dotenv_path=env_path)

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print(f"Warning: GEMINI_API_KEY not found in environment (checked {env_path}).")

# Database Path
DB_PATH = r"c:\SDV\Music\data\datalake\gold\catalog.db"
MODEL_NAME = "gemini-2.5-flash"

# Database schema description for the model
DB_SCHEMA = """
Database Schema:
1. Table 'genres':
   - genre_id: INTEGER PRIMARY KEY
   - track_count: INTEGER (number of tracks)
   - parent: INTEGER (parent genre_id, 0 if top level)
   - title: TEXT (genre name, e.g. "Hip-Hop", "Rock")
   - top_level: INTEGER (top-level genre_id)

2. Table 'artists':
   - artist_id: INTEGER PRIMARY KEY
   - name: TEXT (artist name)
   - location: TEXT (location name)
   - latitude: REAL
   - longitude: REAL
   - bio: TEXT (biography text)

3. Table 'albums':
   - album_id: INTEGER PRIMARY KEY
   - title: TEXT (album title)
   - type: TEXT (e.g. "Album", "Single")
   - date_released: TEXT (YYYY-MM-DD or string)
   - tracks_count: INTEGER (number of tracks in this album)
   - listens: INTEGER (number of listens)
   - favorites: INTEGER (number of favorites)

4. Table 'tracks':
   - track_id: INTEGER PRIMARY KEY
   - album_id: INTEGER (Foreign Key referencing albums)
   - artist_id: INTEGER (Foreign Key referencing artists)
   - title: TEXT (track title)
   - genre_top: TEXT (main top-level genre, e.g. "Hip-Hop", "Rock", "Pop")
   - genres: TEXT (string representation of list of genre IDs, e.g. "[21]")
   - duration: INTEGER (duration in seconds)
   - listens: INTEGER (number of listens)
   - favorites: INTEGER (number of favorites)
   - bit_rate: INTEGER (bit rate in bps)
   - date_created: TEXT (creation date)
"""

def execute_query(sql_query):
    """Executes a SQL query on the gold database and returns results."""
    if not os.path.exists(DB_PATH):
        return None, "Error: Curated database not found."
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        # Format results as a list of dicts
        results = [dict(zip(columns, row)) for row in rows]
        return results, None
    except Exception as e:
        return None, str(e)

def generate_sql(user_question):
    """Uses Gemini API to generate a valid SQLite query for the user's question."""
    if not api_key:
        return None, "API Key missing"
        
    prompt = f"""
You are an expert SQL Generator. Your task is to translate a natural language question into a single valid SQLite query.
{DB_SCHEMA}

Instructions:
1. Generate ONLY a valid SQLite query.
2. Do NOT write any explanations, markdown code blocks, or formatting around the SQL query. Output just the raw SQL text.
3. Be mindful of case-insensitive matches. Use LOWER() or LIKE for text searches where appropriate.
4. Join tables correctly using foreign keys.
5. Limit the results to 10 or 15 unless specified otherwise.
6. Use column names exactly as they are defined.

Question: "{user_question}"
SQL:"""
    
    try:
        # Use gemini-1.5-flash as it is fast and excellent for coding/SQL tasks
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        sql = response.text.strip()
        
        # Clean up code blocks if the model wrapped it anyway
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql, flags=re.IGNORECASE)
        sql = sql.strip("; ") + ";"
        return sql, None
    except Exception as e:
        return None, str(e)

def answer_question(user_question):
    """Orchestrates the Text-to-SQL and final response generation."""
    if not api_key:
        # Local keyword-based fallback if Gemini API is not configured
        return fallback_answer(user_question)
        
    # Step 1: Generate SQL
    sql, err = generate_sql(user_question)
    if err:
        return f"Erreur de génération SQL : {err}", None
        
    # Step 2: Execute SQL
    results, exec_err = execute_query(sql)
    
    # Try self-correction once if SQL fails
    if exec_err:
        correction_prompt = f"""
The following SQLite query failed:
{sql}
Error message: {exec_err}

Generate a corrected, valid SQLite query for the question: "{user_question}".
{DB_SCHEMA}
Output ONLY the raw SQL query, no markdown, no explanations.
"""
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(correction_prompt)
            sql = response.text.strip()
            sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"```\s*", "", sql, flags=re.IGNORECASE)
            sql = sql.strip("; ") + ";"
            results, exec_err = execute_query(sql)
        except Exception:
            pass
            
    if exec_err:
        return f"Erreur d'exécution de la requête SQL : {exec_err}\nRequête générée : `{sql}`", sql

    # Step 3: Formulate final response with results
    final_prompt = f"""
You are the Music Platform Conversational Assistant. You help users query their music catalog.
Answer the user's question using the results of the SQLite database query.

User Question: "{user_question}"
SQL Query Executed: "{sql}"
SQL Query Results: {results}

Instructions:
1. Formulate a friendly and concise response in French.
2. Summarize the key findings from the results.
3. If no results are found, explain it nicely.
4. Format lists or numbers cleanly (e.g. bolding key terms, using bullet points).
5. Do NOT mention details about "database", "SQL", "tables", or "rows" unless the user asked for technical details. Keep it conversational.
"""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(final_prompt)
        return response.text.strip(), sql
    except Exception as e:
        return f"Résultats bruts : {results}\n(Erreur de mise en forme : {e})", sql

def fallback_answer(question):
    """Simple keyword matching fallback if API key is not available."""
    q = question.lower()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        if "combien" in q and "piste" in q:
            cursor.execute("SELECT COUNT(*) FROM tracks;")
            count = cursor.fetchone()[0]
            ans = f"Le catalogue contient actuellement **{count}** pistes musicales."
        elif "artiste" in q and ("plus" in q or "populaire" in q or "écoute" in q):
            cursor.execute("""
                SELECT name, location, COUNT(track_id) as tracks_count, SUM(listens) as total_listens 
                FROM artists JOIN tracks USING (artist_id)
                GROUP BY artist_id ORDER BY total_listens DESC LIMIT 1;
            """)
            row = cursor.fetchone()
            ans = f"L'artiste le plus écouté est **{row[0]}** (basé à {row[1]}) avec un total de **{row[3]:,}** lectures pour {row[2]} morceaux."
        elif "recommande" in q or "conseille" in q:
            cursor.execute("SELECT title, genre_top FROM tracks WHERE genre_top != 'Unknown' ORDER BY listens DESC LIMIT 3;")
            rows = cursor.fetchall()
            recs = "\n".join([f"- **{row[0]}** ({row[1]})" for row in rows])
            ans = f"Voici 3 recommandations populaires du catalogue :\n{recs}"
        else:
            # General fallback
            cursor.execute("SELECT COUNT(*) FROM tracks;")
            tracks_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM artists;")
            artists_count = cursor.fetchone()[0]
            ans = f"Bonjour ! Le catalogue comprend **{tracks_count}** pistes et **{artists_count}** artistes. Posez-moi une question sur le nombre d'écoutes, les genres, les albums ou les artistes !"
        conn.close()
        return ans, "SELECT fallback_query();"
    except Exception as e:
        if conn: conn.close()
        return f"Erreur lors de la recherche locale : {e}", None

if __name__ == "__main__":
    # Test conversational assistant
    print("Testing Assistant (Gemini API)...")
    test_q = "Combien de pistes y a-t-il dans le catalogue ?"
    answer, sql = answer_question(test_q)
    print("Question:", test_q)
    print("SQL Généré:", sql)
    print("Réponse:", answer)
