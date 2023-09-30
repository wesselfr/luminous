use std::fs;
use minifb::{Key, Window, WindowOptions};
use ocl::{
    enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType},
    Context, Device, Image, Kernel, Program, Queue, SpatialDims,
};

const WIDTH: usize = 640;
const HEIGHT: usize = 340;

fn main() {
    let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut time: f32 = 2.0;
    let mut window = Window::new("LUMINOUS", WIDTH, HEIGHT, WindowOptions::default())
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let kernel_source = fs::read_to_string("src/shader.ocl").unwrap();

    let mut program = Program::builder()
        .src(kernel_source)
        .devices(device)
        .build(&context)
        .unwrap();

    let image_buffer = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(SpatialDims::Two(WIDTH, HEIGHT))
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY)
        .queue(queue.clone())
        .build()
        .unwrap();

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Reload
        if window.is_key_released(Key::R) {
            println!("Reload >>");
            let src = fs::read_to_string("src/shader.ocl").unwrap();
            let prog = Program::builder().src(src).devices(device).build(&context);
            match prog {
                Ok(prog) => {
                    program = prog;
                    println!("Reload OK")
                }
                Err(error) => {
                    println!("Reload FAIL");
                    println!("{}", error.to_string())
                }
            }
        }

        // Run OCL Kernel
        unsafe {
            let kernel = Kernel::builder()
                .program(&program)
                .name("main")
                .queue(queue.clone())
                .global_work_size(SpatialDims::Two(WIDTH, HEIGHT))
                .arg(&time)
                .arg(&image_buffer)
                .build()
                .unwrap();
            kernel.enq().unwrap();
        }

        time += 0.1;

        // Copy output back.
        let mut buff = image::ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
            image::Rgba([0, 0, 0, 0])
        });
        image_buffer.read(&mut buff).enq().unwrap();

        // Show output
        for (i, val) in buffer.iter_mut().enumerate() {
            let img = buff.iter().as_slice();

            // Color ARGB
            *val = ((img[i * 4 + 3] as u32) << 24)
                | ((img[i * 4 + 0] as u32) << 16)
                | ((img[i * 4 + 1] as u32) << 8)
                | (img[i * 4 + 2] as u32);
        }

        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}
