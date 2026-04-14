def test_import():
    """Test that the package imports correctly."""
    try:
        import dose_view
        assert hasattr(dose_view, "__all__")
        
        # Test that all main functions are importable
        from dose_view import (
            # I/O
            read_ct_any,
            read_rtdose_file,
            read_dose_image,
            find_rtdose_in_dir,
            get_dose,
            # Resampling
            resample_like,
            resample_to_reference,
            resample_to_common_grid,
            resample_to_spacing,
            # Utils
            window_ct,
            get_image_info,
            print_image_info,
            create_empty_image,
            physical_to_index,
            index_to_physical,
            # Patient
            get_patient_data,
            ct_to_dose,
            dose_to_ct,
            # Visualization
            extract_triplanar_arrays,
            overlay_panel,
            save_triplanar_overlays,
            # Comparison
            extract_line_profile,
            compare_2d_slices,
            compare_1d_profiles,
            compare_depth_dose,
            compute_dose_difference_stats,
        )
        
        print("✓ All imports successful!")
        print(f"✓ Package version: {dose_view.__version__}")
        print(f"✓ Exports {len(dose_view.__all__)} functions")
        return True
        
    except ImportError as e:
        print(f"⚠ Import failed: {e}")
        print("Note: Make sure dependencies are installed:")
        print("  pip install numpy SimpleITK matplotlib")
        print("  pip install pydicom plotly scipy  # for full features")
        return False
    

if __name__ == "__main__":
    import sys
    success = test_import()
    sys.exit(0 if success else 1)


