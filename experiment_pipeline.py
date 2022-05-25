import argparse
from experiment_util import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_prompt", type=str, required=True)
    parser.add_argument("--validate_prompt", type=str, required=True)
    parser.add_argument("--test_prompt", type=str, required=True)
    parser.add_argument("--input", type=str, default="./BeaExperimentSplittedData", required=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--output", type=str, default="./output", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)
    parser.add_argument("--max_norm", type=int, default=10, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    # Step 1. Get args, seed everything and choose device
    args = parse_args()
    seed_everything(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Step 2: Preprocess Data and prepare output folders
    train, train_gold = preprocess(args.input + '/' + args.train_prompt + '/Train', withNER=True)
    validate, validate_gold = preprocess(args.input + '/' + args.validate_prompt + '/Validation', withNER=True)
    test, test_gold = preprocess(args.input + '/' + args.test_prompt + '/Test', withNER=True)

    experiment_name = 'train_' + args.train_prompt.replace('/', '_') + '_epoch_' + str(args.epochs) + '_maxlen_' + str(args.max_len)
    predict_name = experiment_name + "_test_" + str(args.test_prompt)
    model_output_path = args.output + '/' + 'model'
    evaluation_output_path = args.output + '/' + 'evaluation'
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(evaluation_output_path, exist_ok=True)

    prediction_output_path = args.output + '/' + 'prediction'
    os.makedirs(prediction_output_path, exist_ok=True)

    # Step 3: Build Model and Tokenizer
    model, tokenizer = build_model_tokenizer(args.model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    # Step 4: Transfer data
    training_set = FeedbackPrizeDataset(train, tokenizer, args.max_len)
    training_loader = load_data(training_set, args.batch_size)

    # Step 5: Train and Validation
    evaluation_output = {}

    for epoch in range(args.epochs):
        print(f"Training epoch: {epoch + 1}")
        model_train(training_loader, model, optimizer, device, args.max_norm)
        print(f"Validating epoch: {epoch + 1}")
        validate_pred = model_predict(device, model, args.max_len, tokenizer, validate)
        f1, scores = model_evaluate(validate_pred, validate_gold)
        evaluation_output['validation:' + str(epoch + 1)] = scores

    torch.save(model.state_dict(), model_output_path + '/' + experiment_name + '.pt')

    # STEP 5: Test
    print("Test:")
    test_pred = model_predict(device, model, args.max_len, tokenizer, test)
    f1, scores = model_evaluate(test_pred, test_gold)
    evaluation_output['test'] = scores
	
    write_prediction(test_pred, prediction_output_path + '/' + predict_name + '.csv')
    write_evaluation(evaluation_output, evaluation_output_path + '/' + experiment_name + '.csv')
