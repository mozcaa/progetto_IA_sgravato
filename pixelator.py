from PIL import Image

def pixelate_map(input_path, output_path, grid_width):
    """
    Legge un'immagine, la ridimensiona in una griglia per creare un effetto pixelato
    e la salva.
    
    :param input_path: Percorso dell'immagine originale (es. 'mappa.jpg')
    :param output_path: Percorso in cui salvare l'immagine finale (es. 'mappa_pixel.png')
    :param grid_width: Numero di "quadretti" o pixel che comporranno la base
    """
    try:
        # 1. Apri l'immagine originale
        img = Image.open(input_path)
        orig_width, orig_height = img.size
        
        # 2. Calcola l'altezza proporzionale
        # Moltiplichiamo l'altezza originale per il rapporto tra la nuova e la vecchia base
        aspect_ratio = orig_height / orig_width
        grid_height = int(grid_width * aspect_ratio)
        
        # Assicuriamoci che l'altezza sia almeno 1 pixel
        grid_height = max(1, grid_height)
        
        print(f"Dimensioni originali: {orig_width}x{orig_height}")
        print(f"Dimensioni griglia pixel: {grid_width}x{grid_height}")
        
        # 3. Rimpicciolisci l'immagine (Downscale)
        # Usa NEAREST per mappe a colori netti, così non crea colori intermedi sfumati
        # che rovinerebbero i confini degli stati/regioni.
        small_img = img.resize((grid_width, grid_height), resample=Image.Resampling.NEAREST)

        # 4. Salva il risultato
        small_img.save(output_path)
        print(f"Immagine pixelata salvata con successo in: {output_path}")
        
    except FileNotFoundError:
        print(f"Errore: Il file {input_path} non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# ==========================================
# Esempio di utilizzo dello script
# ==========================================
if __name__ == "__main__":
    # Sostituire questi valori con quelli di necessità
    percorso_immagine_originale = "FotoMappa/Originale/mappaItalia1.png"  # Inserisci il nome della tua foto
    percorso_immagine_finale = "FotoMappa/Pixellata/mappaItalia1_pixellata.png"
    
    # parametro_base indica di quante celle deve essere composta la base. L'altezza viene
    # definita in base alla proporzione con la basa per mantenere l'aspect-ratio
    # Più il numero è piccolo, più l'immagine sarà sgranata.
    parametro_base = 200
    
    pixelate_map(
        input_path=percorso_immagine_originale,
        output_path=percorso_immagine_finale,
        grid_width=parametro_base
    )