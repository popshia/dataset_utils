from utils.convert_color import convert_imgs
from utils.draw_yolo_boxes import plot_yolo_labels
from utils.generate_txt import generate_txt
from utils.label_format_converter import convert_label_format
from utils.split_val import split_train_val


class Arguments:
    # generate txt
    train_dir = "\n"
    val_dir = "\n"
    aug = False
    # label format conversion
    input_format = "\n"
    output_format = "\n"
    label_dir = "\n"
    classes_txt = "\n"
    # split train/val
    train_percentage = 0
    # img color conversion
    dataset_dir = "\n"
    # draw yolo labels
    output_count = 0


def script_function_selection():
    function = input(
        """
Please select the function you want to proceed:

1. Generate yolo required train and val txts.
2. Label format conversion.
3. Split train/val sets (w.i.p.).
4. Data augmentation using color conversions.
5. Plot yolo labels and save image results to './save_imgs'.

Your selection: """
    )
    return int(function)


def get_generate_txt_options():
    args = Arguments()
    args.train_dir = input("Path of training directory: ")
    args.val_dir = input("Path of validating directory: ")
    args.aug = True if input("Is the dataset augmented? (y/n) ") == "y" else False
    return args


def get_label_format_convert_options():
    args = Arguments()
    args.input_format = input("Original format of label (xml/txt/json/yaml): ")
    args.output_format = input("New format for label (xml/txt/json/yaml): ")
    args.label_dir = input("Directory of org labels: ")
    args.classes_txt = input("Path of 'classes.txt' file: ")
    return args


def get_split_convert_options():
    args = Arguments()
    args.dataset_dir = input("Dataset dir: ")
    args.train_percentage = int(input("How much percentage is training split? (0-100)"))
    return args


def get_img_color_convert_options():
    args = Arguments()
    args.dataset_dir = input("Dataset dir: ")
    return args


def get_plot_label_options():
    args = Arguments()
    args.dataset_dir = input("Dataset dir: ")
    args.classes_txt = input("Path of 'classes.txt' file: ")
    args.output_count = int(input("How many labels would you like to plot? "))
    return args


if __name__ == "__main__":
    user_selection = script_function_selection()

    match user_selection:
        case 1:
            args = get_generate_txt_options()
            generate_txt(args)
        case 2:
            args = get_label_format_convert_options()
            convert_label_format(args)
        case 3:
            args = get_split_convert_options()
            split_train_val(args)
        case 4:
            args = get_img_color_convert_options()
            convert_imgs(args)
        case 5:
            args = get_plot_label_options()
            plot_yolo_labels(args)
