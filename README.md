    with tab12:
        st.header("🧬 DNA 3D Model Generator")
        
        # Upload file section
        st.subheader("📁 Upload File DNA Sequence")
        uploaded_file = st.file_uploader(
            "Upload file CSV atau Excel berisi sequence DNA", 
            type=['csv', 'xlsx', 'xls'],
            help="File harus mengandung kolom dengan sequence DNA. Format: CSV atau Excel"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Manual Sequence DNA")
            dna_sequence = st.text_area(
                "Masukkan sequence DNA (hanya A, T, C, G):",
                value="ATCGATCGATCGATCGATCG",
                height=100,
                help="Masukkan sequence DNA menggunakan huruf A, T, C, G"
            )
            
            # Validasi input
            dna_sequence = dna_sequence.upper().strip()
            valid_bases = set('ATCG')
            
            if dna_sequence and all(base in valid_bases for base in dna_sequence):
                st.success(f"✅ Sequence valid! Panjang: {len(dna_sequence)} base pairs")
            elif dna_sequence:
                st.error("❌ Sequence mengandung karakter tidak valid. Hanya gunakan A, T, C, G")
        
        with col2:
            st.subheader("Opsi Generasi")
            sequence_length = st.slider(
                "Panjang Sequence (untuk random generator):",
                min_value=10,
                max_value=500,
                value=100,
                help="Pilih panjang sequence untuk generate random DNA"
            )
            
            generate_random = st.button("🎲 Generate Random DNA Sequence")
            
            if generate_random:
                bases = ['A', 'T', 'C', 'G']
                random_sequence = ''.join(np.random.choice(bases, sequence_length))
                dna_sequence = random_sequence
                st.rerun()
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    dna_df = pd.read_csv(uploaded_file)
                else:
                    dna_df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File berhasil diupload! Shape: {dna_df.shape}")
                
                # Cari kolom yang mungkin berisi sequence DNA
                sequence_columns = []
                for col in dna_df.columns:
                    sample_value = str(dna_df[col].iloc[0]) if len(dna_df) > 0 else ""
                    if len(sample_value) > 5 and all(c in 'ATCGatcg' for c in sample_value.upper()):
                        sequence_columns.append(col)
                
                if sequence_columns:
                    selected_column = st.selectbox(
                        "Pilih kolom yang berisi sequence DNA:",
                        sequence_columns
                    )
                    
                    if st.button("🔄 Gunakan Sequence dari File"):
                        dna_sequence = str(dna_df[selected_column].iloc[0])
                        st.rerun()
                
                st.write("Preview data dari file:")
                st.dataframe(dna_df.head())
                
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
        
        # Generate 3D Model dan Analisis
        if dna_sequence and all(base in 'ATCG' for base in dna_sequence.upper()):
            
            # Analisis DNA
            analysis = analyze_dna_sequence(dna_sequence.upper())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Length", analysis['Total Length'])
                st.metric("A Count", analysis['A Count'])
            
            with col2:
                st.metric("T Count", analysis['T Count'])
                st.metric("C Count", analysis['C Count'])
            
            with col3:
                st.metric("G Count", analysis['G Count'])
                st.metric("GC Content", f"{analysis['GC Content']:.2f}%")
            
            with col4:
                st.metric("AT Content", f"{analysis['AT Content']:.2f}%")
            
            # Visualisasi komposisi base
            st.subheader("📊 Komposisi Base DNA")
            
            fig_composition = px.pie(
                values=[analysis['A Count'], analysis['T Count'], analysis['C Count'], analysis['G Count']],
                names=['Adenine (A)', 'Thymine (T)', 'Cytosine (C)', 'Guanine (G)'],
                title='Komposisi Base DNA'
            )
            st.plotly_chart(fig_composition)
            
            # Chart batang untuk komposisi
            fig_bar = px.bar(
                x=['A', 'T', 'C', 'G'],
                y=[analysis['A Count'], analysis['T Count'], analysis['C Count'], analysis['G Count']],
                title='Distribusi Base DNA',
                labels={'x': 'Base', 'y': 'Count'},
                color=['A', 'T', 'C', 'G'],
                color_discrete_map={'A': 'green', 'T': 'yellow', 'C': 'orange', 'G': 'purple'}
            )
            st.plotly_chart(fig_bar)
            
            # Generate 3D Model
            st.subheader("🎨 3D DNA Model Visualization")
            
            with st.spinner("Generating 3D DNA Model..."):
                fig_3d, processed_sequence = generate_dna_3d_model(dna_sequence.upper())
                
                st.plotly_chart(fig_3d)
                
                st.info(f"**Sequence yang digunakan:** {processed_sequence}")
                
                # Kontrol interaktif untuk 3D model
                st.subheader("⚙️ Kontrol Model 3D")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "💾 Download 3D Model (HTML)",
                        data=fig_3d.to_html(),
                        file_name="dna_3d_model.html",
                        mime="text/html"
                    )
                
                with col2:
                    if st.button("🔄 Rotate Model"):
                        st.rerun()
                
                with col3:
                    show_sequence = st.checkbox("Tampilkan Full Sequence", value=False)
                    if show_sequence:
                        st.text_area("Full DNA Sequence:", processed_sequence, height=100)
            
            # Additional analysis
            st.subheader("🔬 Analisis Detail DNA")
            
            # Dinucleotide frequency
            dinucleotides = [f"{dna_sequence[i]}{dna_sequence[i+1]}" for i in range(len(dna_sequence)-1)]
            dinucleotide_counts = pd.Series(dinucleotides).value_counts()
            
            fig_dinucleotide = px.bar(
                x=dinucleotide_counts.index,
                y=dinucleotide_counts.values,
                title='Frekuensi Dinucleotide',
                labels={'x': 'Dinucleotide', 'y': 'Frequency'}
            )
            st.plotly_chart(fig_dinucleotide)
            
            # GC content along sequence (sliding window)
            window_size = min(10, len(dna_sequence) // 10)
            gc_content_window = []
            positions = []
            
            for i in range(0, len(dna_sequence) - window_size + 1, window_size):
                window = dna_sequence[i:i + window_size]
                gc_content = (window.count('G') + window.count('C')) / len(window) * 100
                gc_content_window.append(gc_content)
                positions.append(i)
            
            fig_gc_window = px.line(
                x=positions,
                y=gc_content_window,
                title=f'GC Content (Sliding Window = {window_size} bases)',
                labels={'x': 'Position', 'y': 'GC Content (%)'}
            )
            st.plotly_chart(fig_gc_window)
            
        else:
            st.warning("⚠️ Masukkan sequence DNA yang valid untuk generate model 3D")
            
            # Show example
            with st.expander("🎯 Contoh Format Sequence DNA"):
                st.code("""
CONTOH SEQUENCE DNA YANG VALID:
- ATTCGATCGATCGATCGATCG
- AAAAATTTTTCCCCCGGGGG
- ATCGATCGATCGATCGATCGATCG

ATURAN:
- Hanya gunakan huruf A, T, C, G
- Tidak boleh ada spasi atau karakter khusus
- Case insensitive (A = a, T = t, etc.)
                """)
