            

            # Visualize the matches.
            color = cm.jet(mconf)
            
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text = [], viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text=[])