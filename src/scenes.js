// ============================================================
//  👇 THIS IS THE ONLY FILE YOU EDIT TO CUSTOMISE EVERYTHING.
//
//  Each object = ONE "shot" (one swipe/scroll段).
//  As you scroll through a shot, the photo animates from
//  `start` -> `end`. That's how you control the camera move.
//
//    image  : the photo (put yours in /public/photos/ and
//             use '/photos/your-file.jpg')
//    scale  : zoom. 1 = fit screen, 1.9 = zoomed in a lot
//    x, y   : pan, in %.  negative x = camera pans RIGHT,
//             positive x = pans LEFT.  y = up/down.
//
//  HOW TO GET THE EFFECTS YOU WANTED:
//   • "swipe -> camera goes right along the building"
//        end.x more negative than start.x   (e.g. 0 -> -12)
//   • "swipe -> zoom INTO the window"
//        end.scale big + x/y aimed at the window (e.g. 1.0 -> 1.9, x:18)
//   • "next swipe we're INSIDE (sofas)"
//        make the NEXT shot an interior photo that STARTS zoomed
//        (start.scale 1.6) then pulls back (end.scale 1.0)
//        -> feels like you flew through the window.
//
//  Reuse for a clothing brand tomorrow? Just swap the images
//  + text + numbers below. Same engine. 🎯
// ============================================================

export const scenes = [
  {
    // Replace with: '/photos/exterior-front.jpg'
    image: 'https://picsum.photos/seed/villa-front/1920/1080',
    kicker: 'For Sale',
    title: 'Skyline Villa',
    sub: 'Scroll to take the tour ↓',
    start: { scale: 1.0, x: 0, y: 0 },
    end: { scale: 1.15, x: -10, y: 0 }, // slow drift to the right
  },
  {
    image: 'https://picsum.photos/seed/villa-side/1920/1080',
    kicker: 'Exterior',
    title: 'The West Wing',
    sub: 'Zooming toward the window…',
    start: { scale: 1.05, x: 8, y: 0 },
    end: { scale: 1.9, x: 20, y: 6 }, // ZOOM into the window
  },
  {
    image: 'https://picsum.photos/seed/living-room/1920/1080',
    kicker: 'Interior',
    title: 'Living Room',
    sub: 'Italian sofas, floods of natural light',
    start: { scale: 1.6, x: -6, y: -4 }, // START zoomed = we "entered" the window
    end: { scale: 1.0, x: 0, y: 0 }, // pull back to reveal the room
  },
  {
    image: 'https://picsum.photos/seed/kitchen-modern/1920/1080',
    kicker: 'Interior',
    title: 'Modern Kitchen',
    sub: 'Fully fitted · island counter',
    start: { scale: 1.0, x: -8, y: 0 },
    end: { scale: 1.25, x: 8, y: 0 },
  },
  {
    image: 'https://picsum.photos/seed/bedroom-suite/1920/1080',
    kicker: 'Book a viewing',
    title: 'Yours to own.',
    sub: 'Call +91 00000 00000',
    start: { scale: 1.2, x: 0, y: 0 },
    end: { scale: 1.0, x: 0, y: 0 },
  },
]

// How much scroll each shot takes (higher = slower / more cinematic).
export const SCENE_SCROLL_VH = 130
