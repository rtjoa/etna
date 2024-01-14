From RBT Require Import TypeBasedFuzzer.
From QuickChick Require Import QuickChick.
Set Warnings "-extraction-opaque-accessed,-extraction".
Definition qctest_test_prop_InsertValid := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_InsertValid_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_DeleteValid := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_DeleteValid_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_InsertPost := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_InsertPost_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_DeletePost := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_DeletePost_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_InsertModel := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_InsertModel_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_DeleteModel := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_DeleteModel_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_InsertInsert := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_InsertInsert_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_InsertDelete := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_InsertDelete_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_DeleteInsert := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_DeleteInsert_fuzzer tt))) ++ "}|]")).
Definition qctest_test_prop_DeleteDelete := (fun _ : unit => print_extracted_coq_string ("[|{" ++ show (withTime (fun tt => (test_prop_DeleteDelete_fuzzer tt))) ++ "}|]")).


Parameter OCamlString : Type.
Extract Constant OCamlString => "string".
Axiom qctest_map : OCamlString -> unit.
Extract Constant qctest_map => "
fun test_name ->
  let test_map = [
    (""test_prop_InsertValid"", qctest_test_prop_InsertValid); (""test_prop_DeleteValid"", qctest_test_prop_DeleteValid); (""test_prop_InsertPost"", qctest_test_prop_InsertPost); (""test_prop_DeletePost"", qctest_test_prop_DeletePost); (""test_prop_InsertModel"", qctest_test_prop_InsertModel); (""test_prop_DeleteModel"", qctest_test_prop_DeleteModel); (""test_prop_InsertInsert"", qctest_test_prop_InsertInsert); (""test_prop_InsertDelete"", qctest_test_prop_InsertDelete); (""test_prop_DeleteInsert"", qctest_test_prop_DeleteInsert); (""test_prop_DeleteDelete"", qctest_test_prop_DeleteDelete)
  ] in
  let test = List.assoc test_name test_map in
  test ()


let () =
  Printf.printf ""Entering main of qc_exec\n""; flush stdout;
  setup_shm_aux ();
  Sys.argv.(1) |> qctest_map ; flush stdout;
".

Extraction "TypeBasedFuzzer_test_runner.ml" qctest_test_prop_InsertValid qctest_test_prop_DeleteValid qctest_test_prop_InsertPost qctest_test_prop_DeletePost qctest_test_prop_InsertModel qctest_test_prop_DeleteModel qctest_test_prop_InsertInsert qctest_test_prop_InsertDelete qctest_test_prop_DeleteInsert qctest_test_prop_DeleteDelete qctest_map.