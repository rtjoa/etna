From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 90
      | (0, 3, 5) => 95
      | (0, 3, 6) => 95
      | (0, 5, 3) => 88
      | (0, 5, 5) => 79
      | (0, 5, 6) => 40
      | (0, 6, 3) => 90
      | (0, 6, 5) => 93
      | (0, 6, 6) => 72
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 79
      | (0, 3, 6) => 80
      | (0, 5, 3) => 0
      | (0, 5, 5) => 66
      | (0, 5, 6) => 86
      | (0, 6, 3) => 0
      | (0, 6, 5) => 83
      | (0, 6, 6) => 95
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 31
      | (1, 3, 5) => 65
      | (1, 3, 6) => 79
      | (1, 5, 3) => 33
      | (1, 5, 5) => 15
      | (1, 5, 6) => 40
      | (1, 6, 3) => 55
      | (1, 6, 5) => 51
      | (1, 6, 6) => 58
      | (2, 3, 3) => 4
      | (2, 3, 5) => 34
      | (2, 3, 6) => 8
      | (2, 5, 3) => 56
      | (2, 5, 5) => 34
      | (2, 5, 6) => 43
      | (2, 6, 3) => 1
      | (2, 6, 5) => 25
      | (2, 6, 6) => 3
      | (3, 3, 3) => 46
      | (3, 3, 5) => 46
      | (3, 3, 6) => 46
      | (3, 5, 3) => 0
      | (3, 5, 5) => 26
      | (3, 5, 6) => 0
      | (3, 6, 3) => 44
      | (3, 6, 5) => 54
      | (3, 6, 6) => 44
      | (4, 0, 3) => 14
      | (4, 0, 5) => 0
      | (4, 0, 6) => 16
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 33
      | (1, 3, 6) => 76
      | (1, 5, 3) => 1
      | (1, 5, 5) => 45
      | (1, 5, 6) => 46
      | (1, 6, 3) => 0
      | (1, 6, 5) => 64
      | (1, 6, 6) => 63
      | (2, 3, 3) => 0
      | (2, 3, 5) => 15
      | (2, 3, 6) => 0
      | (2, 5, 3) => 13
      | (2, 5, 5) => 44
      | (2, 5, 6) => 44
      | (2, 6, 3) => 0
      | (2, 6, 5) => 6
      | (2, 6, 6) => 1
      | (3, 3, 3) => 46
      | (3, 3, 5) => 49
      | (3, 3, 6) => 53
      | (3, 5, 3) => 0
      | (3, 5, 5) => 23
      | (3, 5, 6) => 0
      | (3, 6, 3) => 44
      | (3, 6, 5) => 41
      | (3, 6, 6) => 55
      | (4, 0, 3) => 14
      | (4, 0, 5) => 0
      | (4, 0, 6) => 58
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 94
      | (1, 3, 5) => 88
      | (1, 3, 6) => 95
      | (1, 5, 3) => 86
      | (1, 5, 5) => 49
      | (1, 5, 6) => 81
      | (1, 6, 3) => 95
      | (1, 6, 5) => 82
      | (1, 6, 6) => 96
      | (2, 3, 3) => 15
      | (2, 3, 5) => 68
      | (2, 3, 6) => 2
      | (2, 5, 3) => 63
      | (2, 5, 5) => 54
      | (2, 5, 6) => 63
      | (2, 6, 3) => 21
      | (2, 6, 5) => 23
      | (2, 6, 6) => 12
      | (3, 3, 3) => 61
      | (3, 3, 5) => 52
      | (3, 3, 6) => 52
      | (3, 5, 3) => 1
      | (3, 5, 5) => 69
      | (3, 5, 6) => 1
      | (3, 6, 3) => 61
      | (3, 6, 5) => 50
      | (3, 6, 6) => 54
      | (4, 0, 3) => 87
      | (4, 0, 5) => 0
      | (4, 0, 6) => 76
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 96
      | (1, 3, 5) => 82
      | (1, 3, 6) => 92
      | (1, 5, 3) => 57
      | (1, 5, 5) => 78
      | (1, 5, 6) => 17
      | (1, 6, 3) => 94
      | (1, 6, 5) => 82
      | (1, 6, 6) => 91
      | (2, 3, 3) => 96
      | (2, 3, 5) => 75
      | (2, 3, 6) => 96
      | (2, 5, 3) => 62
      | (2, 5, 5) => 64
      | (2, 5, 6) => 49
      | (2, 6, 3) => 96
      | (2, 6, 5) => 88
      | (2, 6, 6) => 95
      | (3, 3, 3) => 46
      | (3, 3, 5) => 54
      | (3, 3, 6) => 49
      | (3, 5, 3) => 96
      | (3, 5, 5) => 66
      | (3, 5, 6) => 96
      | (3, 6, 3) => 49
      | (3, 6, 5) => 54
      | (3, 6, 6) => 47
      | (4, 0, 3) => 20
      | (4, 0, 5) => 96
      | (4, 0, 6) => 27
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
